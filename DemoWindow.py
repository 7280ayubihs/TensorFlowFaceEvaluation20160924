# coding: utf-8
import os
import sys
import Tkinter
from PIL import ImageTk
import matplotlib.pyplot as plot
import DogOrCatEvaluate as doce
import image_process as ip
import take_picture as tp
import take_face_picture as tfp

# カメラから取得する画像のサイズと保存ファイルパス
PICTURE_WIDTH = 300
PICTURE_HEIGHT = 300

# カメラから取得する画像のサイズ
# 幅、または、高さの短い辺を必ず、300にする。
# BSW32KM03（最大解像度 2016×1512ピクセル）の場合は、400x300になる。
CAMERA_VIEW_WIDTH = 400
CAMERA_VIEW_HEIGHT = PICTURE_HEIGHT

# TKINTER関連
TKINTER_BUTTON_CLICK = '<Button-1>'


#
# デモ画面
#
class DemoWindow:
    # 作業用ディレクトリ
    __temp_dir = './temporary'

    # 評価する顔画像ファイルパス
    __temp_face_file = os.path.join(__temp_dir, 'face.png')

    # 結果の円グラフを保存するファイルパス
    __temp_plot_file = os.path.join(__temp_dir, 'plot.png')

    #
    # 作業用ディレクトリに関連する初期化処理
    #
    def __init__temporary(self):
        if not os.path.isdir(self.__temp_dir):
            os.mkdir(self.__temp_dir)

    #
    # 作業用ディレクトリに関連する終了処理
    #
    def __del__temporary(self):
        # テンポラリ画像ファイルが存在する場合は削除
        if os.path.isfile(self.__temp_face_file):
            os.remove(self.__temp_face_file)
        if os.path.isfile(self.__temp_plot_file):
            os.remove(self.__temp_plot_file)

    # 画像のリストと、そのインデックス
    images = []
    images_index = 0

    # 結果イメージのリストと、そのインデックス
    results = []
    results_index = 0

    # 円グラフのラベル
    pie_label = ['Dog', 'Cat']

    #
    # コンストラクタ
    #
    def __init__(self, model_file_path, face_detection=False):
        self.__init__temporary()

        # 顔検出を行うかのフラグ
        self.__face_detection = face_detection

        self.doc = doce.DogOrCatEvaluate(model_file_path)
        plot.rcParams.update({'font.size': 20})

        # ウィンドウの初期設定
        window = Tkinter.Tk()
        window.title('ver.2016.09.24')
        window.geometry('650x460')

        # インフォメーションラベル
        label_info_text = 'TensorFlowに『犬顔』or『猫顔』を評価してもらおう！'
        label_info_font = ('ＭＳ ゴシック', 18, 'bold')
        label_info = Tkinter.Label(text=label_info_text,
                                   font=label_info_font,
                                   width=50)
        label_info.place(x=10, y=10)

        # 結果表示ラベル
        label_result_font = ('ＭＳ ゴシック', 28, 'bold')
        self.label_result = Tkinter.Label(font=label_result_font,
                                          width=32)
        self.label_result.place(x=10, y=400)

        # 画像撮影ボタン
        button_take_picture_text = 'カメラから画像を取得'
        button_take_picture = Tkinter.Button(text=button_take_picture_text,
                                             width=30)
        button_take_picture.bind(TKINTER_BUTTON_CLICK,
                                 self.button_take_picture_clicked)
        button_take_picture.place(x=10, y=50)

        # 評価実行
        button_execute_text = '犬顔 or 猫顔 の評価を実行'
        button_execute = Tkinter.Button(text=button_execute_text,
                                        width=30)
        button_execute.bind(TKINTER_BUTTON_CLICK,
                            self.button_execute_clicked)
        button_execute.place(x=333, y=50)

        # 画像表示キャンバス
        self.canvas_image_view = Tkinter.Canvas(window, width=PICTURE_WIDTH, height=PICTURE_HEIGHT)
        self.canvas_image_view.place(x=10, y=80)
        self.images.append(ImageTk.PhotoImage(file='./resource/no_image.png'))
        self.image_on_canvas = self.canvas_image_view.create_image(150, 150, image=self.images[self.images_index])

        # 結果表示キャンバス
        self.canvas_result_view = Tkinter.Canvas(window, width=PICTURE_WIDTH, height=PICTURE_HEIGHT)
        self.canvas_result_view.place(x=333, y=80)
        self.results.append(ImageTk.PhotoImage(file='./resource/no_image.png'))
        self.result_on_canvas = self.canvas_result_view.create_image(150, 150, image=self.results[self.results_index])

        # ウィンドウを表示
        window.mainloop()

    #
    # デストラクタ
    #
    def __del__(self):
        self.__del__temporary()

    #
    # button_take_picture ボタンのクリックイベント
    # 接続されたカメラから画像を取得し、画像表示キャンバスに取得した画像を表示します。
    #
    def button_take_picture_clicked(self, event):
        # 表示画像を初期化
        self.canvas_image_view.itemconfig(self.image_on_canvas, image=self.images[0])
        self.canvas_result_view.itemconfig(self.result_on_canvas, image=self.results[0])
        self.label_result.config(text='')

        # カメラから画像を取得
        if self.__face_detection:
            tfp.take_face_picture(device_no=1,
                                  size=(CAMERA_VIEW_WIDTH * 2, CAMERA_VIEW_HEIGHT * 2),
                                  save_file=self.__temp_face_file)
        else:
            tp.take_picture(device_no=1,
                            size=(CAMERA_VIEW_WIDTH, CAMERA_VIEW_HEIGHT),
                            save_file=self.__temp_face_file)

        # 画像を取得した場合は、表示
        if os.path.isfile(self.__temp_face_file):
            ip.trimming_square(self.__temp_face_file)
            self.images.append(ImageTk.PhotoImage(file=self.__temp_face_file))
            self.images_index += 1
            self.canvas_image_view.itemconfig(self.image_on_canvas, image=self.images[self.images_index])

    #
    # button_execute_clicked ボタンのクリックイベント
    # カメラから取得した画像を評価し、評価結果を表示します。
    #
    def button_execute_clicked(self, event):
        score = self.doc.evaluate(self.__temp_face_file)
        print ', '.join([self.__temp_face_file, 'dog: ' + str(score[0]), 'cat: ' + str(score[1])])

        # 表示ラベルを更新
        if score[0] < score[1]:
            self.label_result.config(text='この顔は『猫顔』です。')
        else:
            self.label_result.config(text='この顔は『犬顔』です。')

        # 円グラフを作成
        plot.pie(score,
                 pctdistance=0.5,
                 autopct='%1.3f%%',
                 labels=self.pie_label,
                 labeldistance=0.8,
                 colors=['#add8e6', '#90ee90'],
                 startangle=90)
        plot.axis('equal')
        plot.savefig(self.__temp_plot_file)
        plot.close()
        if os.path.isfile(self.__temp_plot_file):
            ip.trimming_square(self.__temp_plot_file)
            ip.resize(self.__temp_plot_file, (PICTURE_WIDTH, PICTURE_HEIGHT))
            self.results.append(ImageTk.PhotoImage(file=self.__temp_plot_file))
            self.results_index += 1
            self.canvas_result_view.itemconfig(self.result_on_canvas, image=self.results[self.results_index])


#
# メイン関数
# ※動作確認用コード
#
if __name__ == "__main__":
    face_detection = False
    for param in sys.argv:
        if param == '-fd':
            face_detection = True

    # 画面を表示
    main_window = DemoWindow('./model/doc2016.ckpt',
                             face_detection=face_detection)
