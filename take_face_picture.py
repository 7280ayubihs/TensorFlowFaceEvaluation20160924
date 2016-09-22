# coding: utf-8
import os
import tkMessageBox
import cv2

# カスケード分類器の情報が書かれているXMLファイルのパス
HAAR_CASCADE_XML_PATH = '/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'


#
# src_img から顔を検出し、最初に検出した顔部分を保存します。
#
# @param src_img 顔検出対象の画像データ
#
# @return rtn_code 検出に成功した場合は、顔部分の画像データを返します。
#                  検出に失敗した場合は、Noneを返します。
#
def take_face(src_img):
    # カスケード分類器の生成
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_XML_PATH)

    # 顔を検出
    face = face_cascade.detectMultiScale(src_img, 1.2, 3)

    # 検出した最初を保存する。
    if 0 < len(face):
        x = face[0][0]
        y = face[0][1]
        width = face[0][2]
        height = face[0][3]
        return src_img[y:y + height, x:x + width]
    else:
        return None


#
# カメラに接続して、’t’ が入力されたら、ファイルに保存する関数
# ...に顔認識機能を追加したもの。
#
# @param device_no カメラデバイスの番号（default: 0）
# @param size 取得するときのサイズ（default: カメラデバイスの最大サイズ）
# @param save_file 保存するファイルパス（default: picture.png）
#
# @return rtn_code 指定された device_no が無効な場合には -1 を返します。
#                     それ以外の場合には、0 を返します。
#
def take_face_picture(device_no=0, size=None, save_file='picture.png'):
    # カメラ接続
    cap = cv2.VideoCapture(device_no)
    if not cap.isOpened():
        return -1

    # 画面に表示
    while True:
        ret, frame = cap.read()

        # リサイズが指定されている場合はリサイズ
        if size is not None and len(size) == 2:
            frame = cv2.resize(frame, size)

        # カメラの映像を表示
        cv2.imshow('take picture', frame)

        # キー入力
        key = cv2.waitKey(1)

        # 't' が入力されたら、画像を保存して終了
        if key == ord('t'):
            face_img = take_face(frame)
            if face_img is not None:
                cv2.imwrite(save_file, face_img)
                break
            else:
                tkMessageBox.showinfo('', '顔を検出できませんでした。' + os.linesep + '再撮影してください。')

        # 'q' が入力されたら、画像を保存せずに終了
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return 0


#a
# メイン関数t
# ※動作確認用コード
#
if __name__ == "__main__":
    # 画像を保存するディレクトリを作成
    SAVE_DIR = "./data/"
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    take_face_picture(1, (800, 600))
