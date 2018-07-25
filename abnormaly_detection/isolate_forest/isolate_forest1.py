from sklearn.ensemble import IsolationForest
import numpy as np
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

y = [1] * len(heights)
y_np = np.array(y)

# 正常値からの学習
clf = IsolationForest()
clf.fit(heights_np, y=y_np)
print("training finished.")

# ----- ここで正常値からの学習終了 ------

# 学習結果を使って異常値かどうかの判断

# テスト用データ
heights_test = [90, 95, 100, 105, 110, 115,  120, 123,  130]
heights_test_np = np.array(heights_test, dtype=np.float32)
heights_test_np = heights_test_np.reshape(-1, 1)

# テスト
results = clf.predict(heights_test_np)

print(results)



