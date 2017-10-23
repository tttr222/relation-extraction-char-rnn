import sys, os, random, pickle, re, time
import numpy as np
import tensorflow as tf
import sklearn.metrics as skm

# 0.001 uniform for embeddings, 0.1 for adagrad accumulators, learning rate 0.1, 0.8 

class CharacterLSTM(object):
    def __init__(self, labels, embedding_size=200, lstm_dim=200, 
                    optimizer='default',
                    learning_rate='default', embedding_factor = 1.0, decay_rate=1.0, 
                    dropout_keep=0.8, num_cores=5):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        config.inter_op_parallelism_threads=num_cores
        config.intra_op_parallelism_threads=num_cores 
        
        self.sess = tf.Session(config=config)
        self.labels = []
        self.embedding_size = embedding_size
        self.optimizer = optimizer
        self.decay = decay_rate
        
        if optimizer == 'default':
            self.optimizer = 'rmsprop'
        else:
            self.optimizer = optimizer
        
        if learning_rate is not 'default':
            self.lrate = float(learning_rate)
        else:
            if self.optimizer in ['adam','rmsprop']:
                self.lrate = 0.001
            elif self.optimizer == 'adagrad':
                self.lrate = 0.5
            else:
                raise Exception('Unknown optimizer {}'.format(optimizer))
        
        print "Optimizer: {}, Learning rate: {}, Decay rate: {}".format(
            self.optimizer, self.lrate, self.decay)
        
        self.embedding_factor = embedding_factor
        self.rnn_dim = lstm_dim
        self.dropout_keep = dropout_keep
        self.char_buckets = 128
        self.labels = labels
        self._compile()
    
    def _compile(self):
        with self.sess.as_default(): 
            import tensorflow_fold as td
        
        output_size = len(self.labels)
        self.keep_prob = tf.placeholder_with_default(tf.constant(1.0),shape=None)

        char_emb = td.Embedding(num_buckets=self.char_buckets, 
                                num_units_out=self.embedding_size)
                                #initializer=tf.truncated_normal_initializer(stddev=0.15))
        char_cell = td.ScopedLayer(tf.contrib.rnn.LSTMCell(num_units=self.rnn_dim), 'char_cell')

        char_lstm = (td.InputTransform(lambda s: [ord(c) for c in s]) 
                    >> td.Map(td.Scalar('int32') >> char_emb) 
                    >> td.RNN(char_cell) >> td.GetItem(1) >> td.GetItem(1))        
        
        rnn_fwdcell = td.ScopedLayer(tf.contrib.rnn.LSTMCell(num_units=self.rnn_dim), 'lstm_fwd')
        fwdlayer = td.RNN(rnn_fwdcell) >> td.GetItem(0)
        
        rnn_bwdcell = td.ScopedLayer(tf.contrib.rnn.LSTMCell(num_units=self.rnn_dim), 'lstm_bwd')
        bwdlayer = (td.Slice(step=-1) >> td.RNN(rnn_bwdcell) 
                        >> td.GetItem(0) >> td.Slice(step=-1))
        
        pos_emb = td.Embedding(num_buckets=300,
                    num_units_out=32,
                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        pos_x = (td.InputTransform(lambda x: x + 150)
                    >> td.Scalar(dtype='int32') 
                    >> pos_emb)
        
        pos_y = (td.InputTransform(lambda x: x + 150)
                    >> td.Scalar(dtype='int32') 
                    >> pos_emb)
        
        input_layer = td.Map(td.Record((char_lstm,pos_x,pos_y)) >> td.Concat())
        
        maxlayer = (td.AllOf(fwdlayer, bwdlayer) 
                    >> td.ZipWith(td.Concat()) 
                    >> td.Max())
        
        output_layer = (input_layer >> 
                        maxlayer >> td.FC(output_size, 
                                         input_keep_prob=self.keep_prob, 
                                         activation=None))

        self.compiler = td.Compiler.create((output_layer, 
                        td.Vector(output_size,dtype=tf.int32)))
                        
        self.y_out, self.y_true = self.compiler.output_tensors
        self.y_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
            logits=self.y_out,labels=self.y_true))

        self.y_prob = tf.nn.softmax(self.y_out)
        self.y_true_idx = tf.argmax(self.y_true,axis=1)
        self.y_pred_idx = tf.argmax(self.y_prob,axis=1)
        
        self.y_pred = tf.one_hot(self.y_pred_idx,depth=output_size,dtype=tf.int32)

        epoch_step = tf.Variable(0, trainable=False)
        self.epoch_step_op = tf.assign(epoch_step, epoch_step+1)
            
        lrate_decay = tf.train.exponential_decay(self.lrate, epoch_step, 1, self.decay)
            
        if self.optimizer == 'adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=lrate_decay)
        elif self.optimizer == 'adagrad':
            self.opt = tf.train.AdagradOptimizer(learning_rate=lrate_decay,
                                                initial_accumulator_value=1e-08)
        elif self.optimizer == 'rmsprop' or self.optimizer == 'default':
            self.opt = tf.train.RMSPropOptimizer(learning_rate=lrate_decay,
                                                 epsilon=1e-08)
        else:
            raise Exception(('The optimizer {} is not in list of available ' 
                            + 'optimizers: default, adam, adagrad, rmsprop.')
                            .format(self.optimizer))
        
        # apply learning multiplier on on embedding learning rate
        embeds = [pos_emb.weights, char_emb.weights]
        grads_and_vars = self.opt.compute_gradients(self.y_loss)
        found = 0
        for i, (grad, var) in enumerate(grads_and_vars):
            if var in embeds:
                found += 1
                grad = tf.scalar_mul(self.embedding_factor, grad)
                grads_and_vars[i] = (grad, var)
        
        assert found == len(embeds)  # internal consistency check
        self.train_step = self.opt.apply_gradients(grads_and_vars)        
        
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=100)
    
    def _onehot(self, y, categories):
        y_onehot = np.zeros((len(y),len(categories)))
        for i in range(len(y)):
            y_onehot[i,categories.index(y[i])] = 1
        
        return y_onehot
    
    def _train_minibatches(self,minibatches):
        mavg_loss = None
        for k, minibatch in enumerate(minibatches):
            varl = [self.train_step, self.y_loss, self.y_pred_idx, self.y_true_idx]
            minibatch[self.keep_prob] = self.dropout_keep
            _, ym_loss, ym_pred, ym_true = self.sess.run(varl, minibatch)
            
            if mavg_loss is None:
                mavg_loss = ym_loss
            else:
                mavg_loss = 0.9 * mavg_loss + 0.1 * ym_loss
            
            sys.stdout.write(" >> training {}/{} loss={:.7f}  \r".format(
                k+1,len(minibatches),mavg_loss))
            sys.stdout.flush()
    
    def fit(self, X, y, X_dev, y_dev, num_epoch = 10, batch_size = 32, seed = 1):
        random.seed(seed)
        print "Target labels: {}".format(self.labels)
        
        trainset = zip(X,self._onehot(y,self.labels))
        devset = zip(X_dev,self._onehot(y_dev,self.labels))
        
        trainsp = random.sample(trainset,min(len(trainset)/2,200))
        trainfd = self.compiler.build_feed_dict(trainsp)
        valfd = self.compiler.build_feed_dict(devset)
        
        best_epoch = 0
        best_model = None
        best_score = 0
        for i in range(1,num_epoch+1):
            estart = time.time()
            batchpool = random.sample(trainset,len(trainset))
            
            minibatches = []
            for k in range(0,len(batchpool),batch_size):
                pool = batchpool[k:k+batch_size]
                minibatches.append(self.compiler.build_feed_dict(pool))
            
            self._train_minibatches(minibatches)
            self.sess.run(self.epoch_step_op)
            
            loss, yt_pred, yt_true = self.sess.run([self.y_loss, self.y_pred_idx, self.y_true_idx], trainfd)
            f1, precision, recall = self.fscore(yt_pred,yt_true)
            
            yv_pred, yv_true = self.sess.run([self.y_pred_idx, self.y_true_idx], valfd)
            vf1, vprecision, vrecall = self.fscore(yv_pred,yv_true)
            
            save_marker = ''
            if vf1 >= best_score:
                best_model = '/tmp/model-{}-e{}-s{}.ckpt'.format(
                    type(self).__name__.lower(),i,seed)
                
                best_epoch, best_score = i, vf1
                self.saver.save(self.sess, best_model)
                save_marker = '*'
                
            elapsed = int(time.time() - estart)
            emin, esec = elapsed / 60, elapsed % 60
            print "epoch {} loss {} fit {:.2f}/{:.2f}/{:.2f} val {:.2f}/{:.2f}/{:.2f} [{}m{}s] {}".format(i, 
                loss, f1, precision, recall, vf1, vprecision, vrecall, emin, esec, save_marker)
        
        if best_model is None:
            print "WARNING: NO GOOD FIT"
        
        self.saver.restore(self.sess, best_model)
        print "Fitted to model from epoch {} with score {} at {}".format(best_epoch,best_score,best_model)
    
    def save(self, model_path):
        self.saver.save(self.sess, model_path)
    
    def restore(self, model_path):
        tf.reset_default_graph()
        self.saver.restore(self.sess, model_path)
    
    def predict(self, X, batch_size = 100):
        dummy_labels = [self.labels[0]] * len(X)
        dummy_y = self._onehot(dummy_labels,self.labels)
        testset_all = zip(X,dummy_y)
        
        prediction_idx = []
        for k in range(0,len(testset_all),batch_size):
            testset = testset_all[k:k+batch_size]
            testfd = self.compiler.build_feed_dict(testset)
            prediction_idx += list(self.sess.run(self.y_pred_idx, testfd))
        
        return [ self.labels[idx] for idx in prediction_idx ]
    
    def predict_proba(self, X, batch_size = 100):
        dummy_labels = [self.labels[0]] * len(X)
        dummy_y = self._onehot(dummy_labels,self.labels)
        testset_all = zip(X,dummy_y)
        
        y_prob_list = []
        for k in range(0,len(testset_all),batch_size):
            testset = testset_all[k:k+batch_size]
            testfd = self.compiler.build_feed_dict(testset)
            y_prob_list.append(self.sess.run(self.y_prob, testfd))
        
        return np.concatenate(y_prob_list,axis=0)

    def evaluate(self,X,y, batch_size = 100, macro = False):
        testset_all = zip(X,self._onehot(y,self.labels))
        
        y_pred_idx = []
        y_true_idx = []
        for k in range(0,len(testset_all),batch_size):
            testset = testset_all[k:k+batch_size]
            testfd = self.compiler.build_feed_dict(testset)
            yp, yt = self.sess.run([self.y_pred_idx,self.y_true_idx], testfd)
            y_pred_idx += list(yp)
            y_true_idx += list(yt)
        
        return self.fscore(y_pred_idx,y_true_idx,macro)
        
    def fscore(self,y_pred,y_true, macro=False):
        avg_meth = 'micro'
        if macro:
            avg_meth = 'macro'
        
        labels = [ i for i in range(len(self.labels)) if self.labels[i] not in ['false','False', False, None, 'other', 'OTHER' ] ]
        f = skm.f1_score(y_true, y_pred, average=avg_meth,labels=labels)
        p = skm.precision_score(y_true, y_pred, average=avg_meth,labels=labels)
        r = skm.recall_score(y_true, y_pred, average=avg_meth,labels=labels)
        return f, p ,r

