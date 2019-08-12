import Algorithmia

client = Algorithmia.client("simSeYQfIQ/XeY+c4pr91rFQQqp1")

input = {
  "video": "data://backSpace001/gsoc/sample2.mp4",
  "detector": "content",
  "output_collection": "data://.algo/media/SceneDetection/temp"
}

result = client.algo("media/SceneDetection/0.1.5").pipe(input).result

print (result)