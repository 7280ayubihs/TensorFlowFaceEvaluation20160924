# coding: utf-8
import os
import datetime as dt
import glob
import random
import cv2
import numpy
import tensorflow as tf
import tensorflow_utility as tu
import model_saver as ms

# 定数
IMAGE_WIDTH = 28    # 画像の幅
IMAGE_HEIGHT = 28   # 画像の高さ
COLOR_CHANNELS = 3  # RGB画像として扱う
IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT * COLOR_CHANNELS


#
# トレーニングを実行し、モデルを保存します。
#
# @param checkpoint_path トレーニングしたモデルを保存するパス
# @param train_data_root_dir 学習データを格納しているディレクトリ
#
def training(checkpoint_path, train_data_root_dir):
    # *************************************************************************
    # 学習データとそのラベルを作成する。
    # *************************************************************************
    # 引数 train_data_root_dir 以下にこのディレクトリが存在するとして、
    # 以降の処理を実施する。
    train_data_dirs = ['dog', 'cat']
    class_num = len(train_data_dirs)

    # 入力データをロード・リサイズしたデータを直列化して学習データとする。
    train_data = []
    train_label = []
    for i, d in enumerate(train_data_dirs):
        files = os.listdir(os.path.join(train_data_root_dir, d))
        for f in files:
            # 画像を読み込み、リサイズする。
            image = cv2.imread(os.path.join(train_data_root_dir, d, f))
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

            # 直列化して学習データとする。
            image = image.flatten().astype(numpy.float32) / 255.0
            train_data.append(image)

            # 学習データに対応するラベルを作成する。
            label = numpy.zeros(class_num)
            label[i] = 1
            train_label.append(label)

    # 学習データとラベルを numpy 配列に変換
    train_data = numpy.asarray(train_data)
    train_label = numpy.asarray(train_label)

    # *************************************************************************
    # ファイルに読み書きするVariableを初期化する
    # *************************************************************************
    # First Convolutional Layer（第1畳み込み層）
    w_conv1 = tu.weight_variable([5, 5, COLOR_CHANNELS, 32], name='w_conv1')
    b_conv1 = tu.bias_variable([32], name='b_conv1')

    # Second Convolutional Layer（第2畳み込み層）
    w_conv2 = tu.weight_variable([5, 5, 32, 64], name='w_conv2')
    b_conv2 = tu.bias_variable([64], name='b_conv2')

    # Densely Connected Layer（全結合層）
    w_fc1 = tu.weight_variable([7 * 7 * 64, 1024], name='w_fc1')
    b_fc1 = tu.bias_variable([1024], name='b_fc1')

    # Readout Leyer（出力層）
    w_fc2 = tu.weight_variable([1024, class_num], name='w_fc2')
    b_fc2 = tu.bias_variable([class_num], name='b_fc2')

    # *************************************************************************
    # 計算途中で使用するVariableを初期化する
    # *************************************************************************
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS])
    y_ = tf.placeholder(tf.float32, shape=[None, class_num])

    # First Convolutional Layer（第1畳み込み層）
    x_image = tf.reshape(x, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_CHANNELS])
    h_conv1 = tf.nn.relu(tu.conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = tu.max_pool_2x2(h_conv1)

    # Second Convolutional Layer（第2畳み込み層）
    h_conv2 = tf.nn.relu(tu.conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = tu.max_pool_2x2(h_conv2)

    # Densely Connected Layer（全結合層）
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    # ドロップアウト層
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout Leyer（出力層）
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

    # *************************************************************************
    # 学習の前準備だけど、何やっているのか不明です。
    # *************************************************************************
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),
                                                  reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # *************************************************************************
    # セッションを生成
    # *************************************************************************
    session = tf.Session()
    init = tf.initialize_all_variables()
    session.run(init)

    # *************************************************************************
    # 学習を実行
    # -->もう少し、効率よく、バッチを作成したい。
    # *************************************************************************
    steps = 200         # 学習ステップ数（default: 100）
    batch_size = 50     # バッチサイズ（default: 50）
    for i in range(steps):
        random_seq = range(len(train_data))
        random.shuffle(random_seq)
        for j in range(len(train_data) / batch_size):
            batch = batch_size * j
            train_data_batch = []
            train_label_batch = []
            for k in range(batch_size):
                train_data_batch.append(train_data[random_seq[batch + k]])
                train_label_batch.append(train_label[random_seq[batch + k]])

            # 学習実行
            train_step.run(session=session,
                           feed_dict={x: train_data_batch,
                                      y_: train_label_batch,
                                      keep_prob: 0.5})

        # 毎ステップ、学習データに対する正答率を表示
        train_accuracy = accuracy.eval(session=session,
                                       feed_dict={x: train_data,
                                                  y_: train_label,
                                                  keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))

    # *************************************************************************
    # 学習したモデルを保存する。
    # *************************************************************************
    save_variables = [w_conv1, b_conv1,
                      w_conv2, b_conv2,
                      w_fc1, b_fc1,
                      w_fc2, b_fc2]
    ms.save(session=session,
            variables=save_variables,
            checkpoint_dir=os.path.dirname(checkpoint_path),
            checkpoint_name=os.path.basename(checkpoint_path))



#
# メイン関数
# ※動作確認用コード
#
if __name__ == "__main__":
    # 前の学習結果を削除
    for f in glob.glob('./model/*'):
        if os.path.isfile(f):
            os.remove(f)

    # 学習実行
    print dt.datetime.today().strftime("%Y-%m-%d %H:%M:%S"), " Training Start"
    training('./model/doc2016.ckpt', './data')
    print dt.datetime.today().strftime("%Y-%m-%d %H:%M:%S"), " Training Finish"
