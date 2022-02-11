# coding: utf-8
# *     SOFTWARE NAME
# *
# *        File:  test.py
# *
# *     Authors: Deleted for purposes of anonymity 
# *
# *     Proprietor: Deleted for purposes of anonymity --- PROPRIETARY INFORMATION
# * 
# * The software and its source code contain valuable trade secrets and shall be maintained in
# * confidence and treated as confidential information. The software may only be used for 
# * evaluation and/or testing purposes, unless otherwise explicitly stated in the terms of a
# * license agreement or nondisclosure agreement with the proprietor of the software. 
# * Any unauthorized publication, transfer to third parties, or duplication of the object or
# * source code---either totally or in part---is strictly prohibited.
# *
# *     Copyright (c) 2019 Proprietor: Deleted for purposes of anonymity
# *     All Rights Reserved.
# *
# * THE PROPRIETOR DISCLAIMS ALL WARRANTIES, EITHER EXPRESS OR 
# * IMPLIED, INCLUDING BUT NOT LIMITED TO IMPLIED WARRANTIES OF MERCHANTABILITY 
# * AND FITNESS FOR A PARTICULAR PURPOSE AND THE WARRANTY AGAINST LATENT 
# * DEFECTS, WITH RESPECT TO THE PROGRAM AND ANY ACCOMPANYING DOCUMENTATION. 
# * 
# * NO LIABILITY FOR CONSEQUENTIAL DAMAGES:
# * IN NO EVENT SHALL THE PROPRIETOR OR ANY OF ITS SUBSIDIARIES BE 
# * LIABLE FOR ANY DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES
# * FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF INFORMATION, OR
# * OTHER PECUNIARY LOSS AND INDIRECT, CONSEQUENTIAL, INCIDENTAL,
# * ECONOMIC OR PUNITIVE DAMAGES) ARISING OUT OF THE USE OF OR INABILITY
# * TO USE THIS PROGRAM, EVEN IF the proprietor HAS BEEN ADVISED OF
# * THE POSSIBILITY OF SUCH DAMAGES.
# * 
# * For purposes of anonymity, the identity of the proprietor is not given herewith. 
# * The identity of the proprietor will be given once the review of the 
# * conference submission is completed. 
# *
# * THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
# *


import sys
import os
import pickle as pkl

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import tensorflow.keras as keras

from netcal.metrics import ECE, ACE


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def weight_variable_glorot(input_dim, output_dim, name=None):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))    
    init = tf.random.uniform(shape=[input_dim, output_dim], minval=-init_range, maxval=init_range, dtype=tf.float32)
    var = tf.Variable(initial_value=init, trainable=True, name=name)
    return var

def dropout_sparse(x, rate, num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    keep_prob = 1. - rate
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(noise_shape)
    retain_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, retain_mask)
    return pre_out * (1./keep_prob)

def masked_accuracy(labels, predicted_quantities, mask):
    correct_prediction = tf.equal(tf.argmax(predicted_quantities, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    
    return tf.reduce_mean(tf.boolean_mask(accuracy_all,mask))


def get_mean_prop_mat(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1)) 
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def get_var_prop_mat(adj, coeff=2.):
    adj_ = sp.coo_matrix(adj)
    rowsum = np.array(adj_.sum(1))   
    degree_vec_inv = np.power(rowsum, -1.).flatten()     
    degree_vec_inv[np.isinf(degree_vec_inv)] = 0.  
    tem = adj_.dot(degree_vec_inv)  
    e = degree_vec_inv*tem/coeff    
    return 1.-e


def get_diag_cdf_loss(labels, m, log_var):
    epsilon = 1e-10
    sigma2 = tf.exp(log_var)
    sigma2_k_star = tf.reduce_sum(tf.math.multiply(labels, sigma2), axis=1, keepdims=True) 
    diag_std = tf.math.sqrt((sigma2_k_star + sigma2)*2.)  
  
    m_k_star = tf.reduce_sum(tf.math.multiply(labels, m), axis=1, keepdims=True)
    delta = m_k_star - m    
    v = delta/diag_std
    
    tem = tf.math.maximum(0.5+tf.math.erf(v)/2.,epsilon)    
    loss = tf.reduce_sum(tf.math.log(tem), axis=1, keepdims=True) 
    loss = - tf.reduce_mean(loss)
    return loss

def get_prediction(m, log_var):  
    sigma2 = tf.exp(log_var)  
    K = m.shape[-1]
    epsilon = 1e-10
    
    tem1 = tf.expand_dims(m, axis=0)
    a = tf.tile(tem1, [K,1,1])
    tem2 = tf.expand_dims(tf.transpose(m), axis=-1)
    b = tf.tile(tem2, [1,1,K])
    diff = b-a
    
    tem1 = tf.expand_dims(sigma2, axis=0)
    a = tf.tile(tem1, [K,1,1])
    tem2 = tf.expand_dims(tf.transpose(sigma2), axis=-1)
    b = tf.tile(tem2, [1,1,K])
    penalty = tf.math.sqrt(2.*(a + b))
    v = diff/penalty
    
    tem = tf.math.maximum(0.5+tf.math.erf(v)/2.,epsilon)
    c = tf.math.log(tem)
    p = tf.transpose(tf.reduce_sum(c,-1))  
    return p

class GraphConvolutionSparse(keras.layers.Layer):
    def __init__(self, input_dim, output_dim, act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        self.act = act
        self.dropout = 0.
        self.features_nonzero = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = weight_variable_glorot(input_dim, output_dim, name=self.name+'_sparsegraphW')

    def call(self, inputs, adj_norm, features_nonzero, dropout=0.,training=None):
        self.dropout = dropout
        self.features_nonzero = features_nonzero
        if training:
            x = dropout_sparse(inputs, dropout, features_nonzero)
        else:
            x = inputs
        x = tf.sparse.sparse_dense_matmul(x, self.W)
        x = tf.sparse.sparse_dense_matmul(adj_norm, x)
        return self.act(x)
    
    def get_l2_reg(self):
        self.l2_reg = tf.nn.l2_loss(self.W)
        return self.l2_reg
    
class GraphConvolution(keras.layers.Layer):    
    def __init__(self, input_dim, output_dim, act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.act = act
        self.dropout = 0.
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = weight_variable_glorot(input_dim, output_dim, name=self.name+'_graphW')

    def call(self, inputs, adj_norm, dropout=0.,training=None):
        self.dropout = dropout
        if training:
            x = tf.nn.dropout(inputs, rate=dropout)
        else:
            x = inputs            
        
        x = tf.matmul(x, self.W)
        x = tf.sparse.sparse_dense_matmul(adj_norm, x)
        return self.act(x)
    
    def get_l2_reg(self):
        self.l2_reg = tf.nn.l2_loss(self.W)
        return self.l2_reg
    
class GraphConvolutionSparseVar(keras.layers.Layer):
    def __init__(self, input_dim, output_dim, act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparseVar, self).__init__(**kwargs)
        self.act = act
        self.dropout = 0.
        self.features_nonzero = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = weight_variable_glorot(input_dim, output_dim, name=self.name+'_sparsegraphVarW')

    def call(self, inputs, adj_var, features_nonzero, dropout=0.,training=None):
        self.dropout = dropout
        self.features_nonzero = features_nonzero
        if training:
            x = dropout_sparse(inputs, dropout, features_nonzero)
        else:
            x = inputs
        
        x = tf.sparse.sparse_dense_matmul(x, self.W)
        x = tf.transpose(tf.multiply(tf.transpose(x), adj_var))
        return self.act(x)
    
    def get_l2_reg(self):
        self.l2_reg = tf.nn.l2_loss(self.W)
        return self.l2_reg
    
class GraphConvolutionVar(keras.layers.Layer):    
    def __init__(self, input_dim, output_dim, act=tf.nn.relu, **kwargs):
        super(GraphConvolutionVar, self).__init__(**kwargs)
        self.act = act
        self.dropout = 0.
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = weight_variable_glorot(input_dim, output_dim, name=self.name+'_graphVarW')

    def call(self, inputs, adj_var, dropout=0.,training=None):
        self.dropout = dropout
        if training:
            x = tf.nn.dropout(inputs, rate=dropout)
        else:
            x = inputs            
        
        x = tf.matmul(x, self.W)
        x = tf.transpose(tf.multiply(tf.transpose(x), adj_var))
        return self.act(x)
    
    def get_l2_reg(self):
        self.l2_reg = tf.nn.l2_loss(self.W)
        return self.l2_reg    

class NodeUncertainty(keras.layers.Layer):
    def __init__(self, num_features, hidden_dim, output_dim, **kwargs):
        super(NodeUncertainty, self).__init__(**kwargs)
        
        self.input_dim = num_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.features_nonzero = None                
                  
        self.hidden1_mean = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=self.hidden_dim[0],
                                              act=tf.nn.relu)
        self.z_mean = GraphConvolution(input_dim=self.hidden_dim[0],
                                       output_dim=self.output_dim,
                                       act=lambda x: x)
        
        self.hidden1_log_var = GraphConvolutionSparseVar(input_dim=self.input_dim,
                                              output_dim=self.hidden_dim[0],
                                              act=tf.nn.relu)        
        self.z_log_var = GraphConvolutionVar(input_dim=self.hidden_dim[0],
                                          output_dim=self.output_dim,
                                          act=lambda x: x)
        
        
    def call(self, inputs, adj_norm, adj_var, features_nonzero,dropout=[0.,0.],training=None):
        self.features_nonzero = features_nonzero
        
        x = self.hidden1_mean(inputs,adj_norm,features_nonzero,dropout[0],training)
        m = self.z_mean(x,adj_norm,dropout[1],training)        
        x_var = self.hidden1_log_var(inputs,adj_var,features_nonzero,dropout[0],training)
        log_var = self.z_log_var(x_var,adj_var,dropout[1],training)
        
        return m, log_var
    
    def get_l2_reg(self):
        l2_reg_layer0_mean = self.hidden1_mean.get_l2_reg()
        l2_reg_layer0_var = self.hidden1_log_var.get_l2_reg()
        l2_reg_layer1_mean = self.z_mean.get_l2_reg()
        l2_reg_layer1_var = self.z_log_var.get_l2_reg()
        return l2_reg_layer0_mean, l2_reg_layer0_var, l2_reg_layer1_mean, l2_reg_layer1_var
    
    
def run(dataset_str = 'cora',
        rerun_id = 0,
        train_perk = 10,
        hidden_dim = [64],
        weight_decay = 1e-3,
        learning_rate = 0.0005,
        epochs = 2000,
        dropout = [0.2,0.2],
        patience = 20):

    if not os.path.exists('results'):
        os.makedirs('results')    
    output_path = 'results/up_' + dataset_str + '_rerun' + str(rerun_id) + '_' + str(train_perk) + 'perk'   
    output_pickle = output_path + '/res.pkl'
    
    fn = 'data/' + dataset_str + '_features.pkl'
    with open(fn, 'rb') as f:
        features = pkl.load(f)
    num_nodes = features.shape[0]
    num_features = features.shape[1]
    features_nonzero = features.count_nonzero()
    features = sparse_to_tuple(features.tocoo())

    fn = 'data/' + dataset_str + '_adj.pkl'
    with open(fn, 'rb') as f:
        adj = pkl.load(f)
    
    fn = 'data/'+dataset_str+'_rerun'+str(rerun_id)+'_val.npz'
    with np.load(fn) as tem:
        y_val = tem['y_val'].astype('float32')
        val_mask = tem['val_mask']
    y_val_sq = y_val[val_mask]
    
    fn = 'data/'+dataset_str+'_rerun'+str(rerun_id)+'_test.npz'
    with np.load(fn) as tem:
        y_test=tem['y_test'].astype('float32')  
        test_mask=tem['test_mask']  
    y_test_sq = y_test[test_mask]
    
    fn = 'data/'+dataset_str+'_rerun'+str(rerun_id)+'_train_'+str(train_perk)+'perk.npz'
    with np.load(fn) as tem:
        y_train=tem['y_train'].astype('float32')
        train_mask=tem['train_mask']
    y_train_sq = y_train[train_mask]
    num_classes = y_train.shape[1]
    
    features_tfs = tf.sparse.SparseTensor(features[0],np.float32(features[1]),features[2])
    adj_norm = get_mean_prop_mat(adj)
    adj_norm_tfs = tf.sparse.SparseTensor(adj_norm[0],np.float32(adj_norm[1]),adj_norm[2])
    adj_var = get_var_prop_mat(adj)

    model = NodeUncertainty(num_features, hidden_dim, num_classes)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=model)
    manager = tf.train.CheckpointManager(ckpt, output_path, max_to_keep=3)

    acc_val_max = 0.
    loss_val_min = np.inf
    curr_step = 0
    for epoch in range(epochs):
        ckpt.step.assign_add(1)

        with tf.GradientTape() as tape:
            m, log_var = model(features_tfs, adj_norm_tfs, adj_var, features_nonzero, dropout,training=True)
            loss_data = get_diag_cdf_loss(y_train_sq, tf.boolean_mask(m, train_mask), tf.boolean_mask(log_var, train_mask))  
            l2_reg_layer0_mean, l2_reg_layer0_var, l2_reg_layer1_mean, l2_reg_layer1_var= model.get_l2_reg()
            loss = loss_data + weight_decay * (l2_reg_layer0_mean + l2_reg_layer0_var + l2_reg_layer1_mean + l2_reg_layer1_var)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))     

        m_testmode, log_var_testmode = model(features_tfs, adj_norm_tfs, adj_var, features_nonzero, dropout,training=False)
        pred = get_prediction(m_testmode, log_var_testmode)
        acc_train = masked_accuracy(y_train, pred, train_mask)
        acc_test = masked_accuracy(y_test, pred, test_mask)
        acc_val = masked_accuracy(y_val, pred, val_mask)
        
        loss_data_val = get_diag_cdf_loss(y_val_sq, tf.boolean_mask(m_testmode, val_mask), tf.boolean_mask(log_var_testmode, val_mask)) 
        loss_val = loss_data_val + weight_decay * (l2_reg_layer0_mean + l2_reg_layer0_var + l2_reg_layer1_mean + l2_reg_layer1_var)         
            
        if acc_val >= acc_val_max or loss_val <= loss_val_min:
            if acc_val >= acc_val_max and loss_val <= loss_val_min:
                acc_val_early = acc_val.numpy()
                loss_val_early = loss_val.numpy()
                check_path = manager.save()
            acc_val_max = np.max((acc_val.numpy(),acc_val_max))
            loss_val_min = np.min((loss_val.numpy(),loss_val_min))
            curr_step = 0
        else:
            curr_step += 1
            if curr_step == patience:
                break
                
    ckpt.restore(manager.latest_checkpoint)    
    m_testmode, log_var_testmode = model(features_tfs, adj_norm_tfs, adj_var, features_nonzero, dropout,training=False)
    pred = get_prediction(m_testmode, log_var_testmode)
    acc_test = masked_accuracy(y_test, pred, test_mask)
    
    test_pred = np.nan_to_num(tf.exp(tf.boolean_mask(pred,test_mask)/(num_classes-1.)).numpy()) 
    test_true = np.argmax(y_test_sq,axis=1)
    n_bins = 20
    ace = ACE(n_bins).measure(test_pred, test_true) * 100
    ece = ECE(n_bins).measure(test_pred, test_true) * 100

    meta_info = {}
    meta_info['dataset_str'] = dataset_str
    meta_info['hidden_dim'] = hidden_dim
    meta_info['weight_decay'] = weight_decay
    meta_info['learning_rate'] = learning_rate
    meta_info['epochs'] = epochs
    meta_info['dropout'] = dropout
    meta_info['patience'] = patience
    meta_info['rerun_id'] = rerun_id
    meta_info['train_perk'] = train_perk
    meta_info['acc'] = acc_test.numpy()
    meta_info['ece'] = ece
    meta_info['ace'] = ace  
    
    with open(output_pickle, 'wb') as f:
        pkl.dump(meta_info, f, protocol=pkl.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    dataset_str = sys.argv[1] 
    rerun_id = int(sys.argv[2])
    train_perk = int(sys.argv[3])
    
    hidden_dim = [64]
    weight_decay = 1e-3
    learning_rate = 0.0005
    epochs = 2000
    dropout = [0.2,0.2]
    patience = 20
    
    run(dataset_str,
        rerun_id,
        train_perk,
        hidden_dim,
        weight_decay, 
        learning_rate,
        epochs,
        dropout,
        patience)