import onnx

class onnx_bk:
    def __init__(self, onnx):
        """
        onnxモデルからバックエンド用のオブジェクトを作る
        :param onnx: ModelProto
        """
        self.model = onnx

    def get_all_layer_names(self):
        """
        全てのレイヤー名をリストにして返す
        :return:
        """
        all_layer_names = []
        for _, op in enumerate(self.model.graph.node):
            all_layer_names.append(op)

        return all_layer_names

    def get_input_shape(self,  layer_name):
        """
        レイヤー名から入力のテンソルの形を取り出す
        :param layer_name:
        :return: tensor
        """
        # 一つ前のレイヤーの情報を取得
        prev_layer = self.get_prev_layer(layer_name)

        return prev_layer.ouput.shape

