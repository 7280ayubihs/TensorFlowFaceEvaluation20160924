# coding: utf-8
import os
import cv2


#
# 画像を短い辺に合わせて正方形にします。
#
# @param image_file_path - 正方形に変形する画像ファイル
#
def trimming_square(image_file_path):
    img = cv2.imread(image_file_path, 1)

    # タテ、ヨコ、チャンネル数を取得
    height, width, channels = img.shape

    # 正方形にトリミング
    if height < width:
        # 横長の場合
        n = (width - height) / 2
        img = img[0:height, n:n + height]
    elif height > width:
        # 縦長の場合
        n = (height - width) / 2
        img = img[n:n + width, 0:width]

    # 上書き保存
    cv2.imwrite(image_file_path, img)


#
# 画像指定サイズにリサイズします。
#
# @param image_file_path - 正方形に変形する画像ファイル
# @param size            - サイズ
#
def resize(image_file_path, size):
    img = cv2.imread(image_file_path, 1)
    img = cv2.resize(img, size)

    # 上書き保存
    cv2.imwrite(image_file_path, img)


#
# src_dir にある画像に以下の処理をして dst_dir に保存します。
#   1. 正方形にトリミング
#   2. 128x128にリサイズ（サイズ指定可能）
#   3. グレースケール化（処理しないことも可能）
#   4. PNG形式に変換
#
# @param src_dir 処理する画像が格納されたディレクトリ
# @param dst_dir 処理した画像を保存するディレクトリ
# @param size リサイズ後のサイズ（default: 128x128）
# @param grey_scale Trueの時にグレースケール処理を実施（default: True）
#
# @return なし
def image_process(src_dir, dst_dir, size=(128, 128), grey_scale=True):
    for image_file in os.listdir(src_dir):
        # 読み込み
        img = cv2.imread(os.path.join(src_dir, image_file), 1)
        if img is None:
            continue

        # タテ、ヨコ、チャンネル数を取得
        height, width, channels = img.shape

        # 正方形にトリミング
        if height < width:
            # 横長の場合
            n = (width - height) / 2
            img = img[0:height, n:n + height]
        elif height > width:
            # 縦長の場合
            n = (height - width) / 2
            img = img[n:n + width, 0:width]

        # 128x128へのリサイズ
        img = cv2.resize(img, size)

        # グレースケール化
        if grey_scale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # PNG形式で保存
        cv2.imwrite(os.path.join(dst_dir, image_file + '.png'), img)


#
# メイン関数
# ※動作確認用コード
#
if __name__ == "__main__":
    # ファイルリストを取得
    image_process('./src', './dst')
    # image_process('./src', './dst', (128, 128), True)
    # image_process('./src', './dst', (64, 64), True)
    # image_process('./src', './dst', (128, 128), False)
