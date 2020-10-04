import paddle.fluid as fluid

def reader():
    for i in range(10):
        yield i
        
batch_reader = fluid.io.batch(reader, batch_size=2)

for data in batch_reader():
    print(data)