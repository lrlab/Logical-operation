# coding: utf-8
import numpy as np
from sklearn.cross_validation import train_test_split
from collections import defaultdict
import six
import sys
import chainer
import chainer.links as L
from chainer import optimizers, cuda, serializers
import chainer.functions as F
import argparse
from gensim import corpora, matutils

"""
Multilayer perceptron

多層パーセプトロンでORを近似
 - 入力層2ユニット
 - 出力層1ユニット 

 回帰問題としてORを学習する。
 予測する時は閾値を設ける必要あり。
 @author ichiroex
"""

#引数の設定
parser = argparse.ArgumentParser()
parser.add_argument('--gpu  '    , dest='gpu'        , type=int, default=0,              help='1: use gpu, 0: use cpu')
parser.add_argument('--epoch'    , dest='epoch'      , type=int, default=50000,          help='number of epochs to learn')

args = parser.parse_args()
n_epoch     = args.epoch        # エポック数(パラメータ更新回数)

# Prepare dataset
source = [[0, 0], [1, 0], [0, 1], [1, 1]]
target = [[0], [1], [1], [1]]
dataset = {}
dataset['source'] = np.array(source, dtype=np.float32)
dataset['target'] = np.array(target, dtype=np.float32)

N = len(source) # train data size

in_units  = 2   # 入力層のユニット数
out_units = 1   # 出力層のユニット数

#モデルの定義
model = chainer.Chain(l1=L.Linear(in_units, out_units))

#GPUを使うかどうか
if args.gpu > 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    xp = np if args.gpu <= 0 else cuda.cupy #args.gpu <= 0: use cpu, otherwise: use gpu

### 変更
else:
    xp = np

def forward(x, t):
    return F.sigmoid(model.l1(x))

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

f = open('loss_or.dat', 'w')

# Learning loop
loss_val = 100
epoch = 0
while loss_val > 1e-5:

    # training
    x = chainer.Variable(xp.asarray(dataset['source'])) #source
    t = chainer.Variable(xp.asarray(dataset['target'])) #target
    
    model.zerograds()       # 勾配をゼロ初期化
    y    = forward(x, t)    # 順伝搬

    loss = F.mean_squared_error(y, t) #平均二乗誤差
    
    loss.backward()              # 誤差逆伝播
    optimizer.update()           # 最適化

    
    # 途中結果を表示
    if epoch % 1000 == 0:
        #誤差と正解率を計算
        loss_val = loss.data
        
        print 'epoch:', epoch
        print 'x:\n', x.data
        print 't:\n', t.data
        print 'y:\n', y.data
        print 'l1.W:\n', model.l1.W.data
        print 'l1.b:\n', model.l1.b.data
        print('train mean loss={}'.format(loss_val)) # 訓練誤差, 正解率
        print ' - - - - - - - - - '
    
    f.write(str(loss.data)+'\n')
    # n_epoch以上になると終了
    if epoch >= n_epoch:
        break

    epoch += 1

print 'save the loss.dat'
f.close()

#modelとoptimizerを保存
print 'save the model'
serializers.save_npz('or_mlp.model', model)
print 'save the optimizer'
serializers.save_npz('or_mlp.state', optimizer)

