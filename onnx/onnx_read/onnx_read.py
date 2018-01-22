import onnx

onnx_file = 'squeezenet.onnx'

model = onnx.ModelProto()
with open(onnx_file, 'rb') as fp:
    content = fp.read()
    model.ParseFromString(content)


print(dir(model.graph))
inputs = model.graph.input


for op_id, op in enumerate(model.graph.node):
    print(op_id, " ", op)

for input in inputs:
    print(input)

init = model.graph.initializer

for initializer in model.graph.initializer:
    if initializer.name == "2":
        print(initializer)
        data = initializer.raw_data
        print(len(data))

print('finished')


