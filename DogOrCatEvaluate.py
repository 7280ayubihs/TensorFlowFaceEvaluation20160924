# coding: utf-8
import os
import cv2
import numpy
import tensorflow as tf
import model_saver as ms
import tensorflow_utility as tfu


#
# 犬顔 or 猫顔の評価を行う機能を提供するクラス
#
class DogOrCatEvaluate:
    #
    # コンストラクタ
    #
    # @param checkpoint_path - 学習済みのモデルファイルのパス
    #
    def __init__(self, checkpoint_path):
        # 評価画像の設定
        self.img_width = 28      # 画像の幅
        self.img_height = 28     # 画像の高さ
        self.color_channels = 3  # カラーチャンネル（GrayScale: 1, RGB: 3）
        self.img_pixels = self.img_width * self.img_height * self.color_channels

        # クラス変数の初期化
        self.__eval_class = ['dog', 'cat']
        self.class_num = len(self.__eval_class)

        self.__init_tf_variable()

        # モデルを復元
        self.session = tf.Session()
        ms.restore(self.session,
                   os.path.dirname(checkpoint_path),
                   os.path.basename(checkpoint_path))

    #
    # デストラクタ
    #
    def __del__(self):
        self.session.close()

    #
    # TensorFlow で使用する変数を初期化します。
    #
    def __init_tf_variable(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.img_pixels])
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.class_num])

        # First Convolutional Layer（第1畳み込み層）
        self.w_conv1 = tfu.weight_variable([5, 5, self.color_channels, 32], name='w_conv1')
        self.b_conv1 = tfu.bias_variable([32], name='b_conv1')
        x_image = tf.reshape(self.x, [-1, self.img_width, self.img_height, self.color_channels])
        h_conv1 = tf.nn.relu(tfu.conv2d(x_image, self.w_conv1) + self.b_conv1)
        h_pool1 = tfu.max_pool_2x2(h_conv1)

        # Second Convolutional Layer（第2畳み込み層）
        self.w_conv2 = tfu.weight_variable([5, 5, 32, 64], name='w_conv2')
        self.b_conv2 = tfu.bias_variable([64], name='b_conv2')
        h_conv2 = tf.nn.relu(tfu.conv2d(h_pool1, self.w_conv2) + self.b_conv2)
        h_pool2 = tfu.max_pool_2x2(h_conv2)

        # Densely Connected Layer（全結合層）
        self.w_fc1 = tfu.weight_variable([7 * 7 * 64, 1024], name='w_fc1')
        self.b_fc1 = tfu.bias_variable([1024], name='b_fc1')
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.w_fc1) + self.b_fc1)

        # ドロップアウト層
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Readout Leyer（出力層）
        self.w_fc2 = tfu.weight_variable([1024, self.class_num], name='w_fc2')
        self.b_fc2 = tfu.bias_variable([self.class_num], name='b_fc2')
        self.y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, self.w_fc2) + self.b_fc2)

    #
    # 犬顔 or 猫顔の評価を行います。評価結果は、[犬顔%,猫顔%]の配列で返されます。
    #
    # @param eval_image_path - 評価する画像のパス
    # @return [犬顔,猫顔]のfloat型の配列を返します。
    #         評価に失敗した場合は、[-1.0, -1.0] の配列を返します。
    #
    def evaluate(self, eval_image_path):
        # 評価データとそのラベルを作成する。
        eval_file, eval_data, eval_label = self.__convert_evaluate_image(eval_image_path)
        if eval_file is None:
            return [-1.0, -1.0]

        # 画像を評価し、スコアを返す。
        score = self.session.run(self.y_conv,
                                 feed_dict={self.x: [eval_data],
                                            self.y_: [eval_label],
                                            self.keep_prob: 1.0})
        return score[0]

    #
    # 評価する画像を読み込み、TensorFlowで評価可能なデータ構造に変換する。
    #
    # @param eval_image_dir - 評価する画像が格納されているディレクトリ
    #
    def __convert_evaluate_image(self, eval_image_file):
        # 入力データをロード・リサイズしたデータを直列化して学習データとする。
        eval_file = []
        eval_data = []
        eval_label = []

        image = cv2.imread(eval_image_file)
        if image is None:
            return None, None, None

        # 評価画像名を追加
        eval_file.append(eval_image_file)

        # 画像をリサイズ
        image = cv2.resize(image, (self.img_width, self.img_height))

        # 直列化して学習データとする。
        image = image.flatten().astype(numpy.float32) / 255.0
        eval_data.append(image)

        # 学習データに対応するラベルを作成する。
        label = numpy.zeros(self.class_num)
        eval_label.append(label)

        # 読み込んだファイル名と、評価データとラベルを numpy 配列に変換したものを返す。
        return (eval_file[0],
                numpy.asarray(eval_data[0]),
                numpy.asarray(eval_label[0]))


#
# メイン関数
# ※動作確認用コード
#
if __name__ == "__main__":
    # 評価用画像のディレクトリ
    # EVAL_IMAGE_DIR = './data/human'
    EVAL_IMAGE_DIR = './data/div32'

    # 評価モデルを作成
    doc = DogOrCatEvaluate('./model/doc2016.ckpt')

    # 評価する画像のリストを作成
    eval_images = os.listdir(EVAL_IMAGE_DIR)
    for eval_image in eval_images:
        score = doc.evaluate(os.path.join(EVAL_IMAGE_DIR, eval_image))
        print ','.join([eval_image, 'dog:' + str(score[0]), 'cat:' + str(score[1])])
