from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt
# 元データ
# https://www.e-stat.go.jp/stat-search/files?page=1&query=%E4%BD%93%E9%87%8D&layout=dataset&stat_infid=000031557631

rng = np.random.RandomState(42)

# 1変数によるisolation forest
# 身長の分布（正常値）
# 実際はcsvから読み込んでください
heights = [102, 102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 105, 105, 105, 106, 106, 106, 106, 106, 106, 106,
           106, 106, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 108, 108, 108, 108, 108, 108, 108, 108, 108,
           108, 108, 108, 108, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 110, 110, 110, 110, 110, 110,
           110, 110, 110, 110, 110, 110, 110, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 112,
           112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
           113, 113, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 115, 115, 115, 115, 115, 115, 116, 116, 116,
           116, 117, 118]

weights = [ 15, 16, 15, 16, 17, 15, 16, 17, 16, 16, 17, 17, 17, 18, 16, 16, 16, 17, 17, 17, 18, 18, 19, 16, 17, 17,
            17, 17, 18, 18, 18, 18, 19, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19 , 19 , 19 , 20, 16, 17, 17, 18, 18,
            18, 18, 19, 19, 19, 20, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 21, 17, 17, 18, 18, 18, 19, 19,
            19, 19, 20, 20, 20, 21, 17, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 21, 18, 18, 18, 19, 19, 19, 20, 20,
            20, 21, 21, 22, 18, 19, 19, 19, 20, 20, 20, 21, 21, 22, 18, 19, 19, 20, 20, 21, 18, 19, 20, 21, 21, 21]


# numpyに変換
heights_np = np.array(heights, dtype=np.float32).reshape(-1, 1)
weights_np = np.array(weights, dtype=np.float32).reshape(-1, 1)
# 2次元へ変換
train_np = np.c_[heights_np, weights_np]

# 正常値からの学習
# contamination が全データに含まれる異常値の割合。
# データの中に10%の異常が含まれる場合、0.1を指定する。
# 何も指定しないときは、contamination = 0.1
clf = IsolationForest(contamination=0)
clf.fit(train_np)
print("training finished.")

# ----- ここで正常値からの学習終了 ------

# 学習結果を使って異常値かどうかの判断
# テスト用データ
# テスト
# results = clf.predict(train_np)


# 身長と体重のペアを作る。次のグラフ描画で使用する。
# 身長について90から120まで50個、体重については10から30までの50個。
# 身長と体重の組み合わせは2500個
xx, yy = np.meshgrid(np.linspace(90, 120, 50), np.linspace(10, 30, 50))

# xx_yyの形は (2500, 2) 2500は身長と体重のペア、2が特徴量（身長、体重）
xx_yy = np.c_[xx.ravel(), yy.ravel()]
# 身長、体重の組み合わせについて、全ての異常値を計算して、グラフに表示する。
Z = clf.decision_function(xx_yy)
Z = Z.reshape(xx.shape)

plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
plt.savefig('plot2d.png')
plt.show()

# ある身長と体重の組み合わせが以上かどうか判断
test_data = [[100, 25],   [110, 25],  [120, 25]]

data = np.array(test_data)
results = clf.predict(data)

for d, r in zip(test_data, results):
    if r == 1:
        judge = '正常'
    else:
        judge = '異常'

    print("判定 %s, 身長 %d, 体重 %d" % (judge, d[0], d[1]))
