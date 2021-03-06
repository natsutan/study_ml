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
           116, 117, 118, ]

# numpyに変換
heights_np = np.array(heights, dtype=np.float32)

# 1次元へ変換
heights_np = heights_np.reshape(-1, 1)

# 正常値からの学習
# contamination が全データに含まれる異常値の割合。
# データの中に10%の異常が含まれる場合、0.1を指定する。
# 何も指定しないときは、contamination = 0.1
contamination = 0.1
clf = IsolationForest(contamination=contamination)
clf.fit(heights_np)
print("training finished.")

# ----- ここで正常値からの学習終了 ------
# 学習結果を使って異常値かどうかの判断

# テスト
# decision_functionの代わりにpredictを使うと、数値ではなく異常かどうかだけを返してくれる
test_np = np.arange(95, 125, 1).reshape(-1, 1)
results = clf.decision_function(test_np)

# 値が小さいほど異常値とみなせる
b1 = plt.scatter(test_np, results, c='white', s=20, edgecolor='k')
filename = "plot_" + str(contamination) + '.png'
plt.savefig(filename)
plt.show()




