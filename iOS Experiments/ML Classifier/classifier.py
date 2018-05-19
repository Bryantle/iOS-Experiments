import turicreate as turi

url = "dataset/"

data = turi.image_analysis.load_images(url)

data['foodtype'] = data['path'].apply(lambda path: "Rice" if "rice" in path else "Soup")

data.save("rice_or_soup.sframe")


dataBuffer = turi.SFrame("rice_or_soup.sframe")

trainingBuffers, testingBuffers = dataBuffer.random_split(0.9)

model = turi.image_classifier.create(trainingBUffers, target = "foodtype", model="resnet-50")

evaluations = model.evaluate(testingBuffers)

print (evaluations["accuracy"])


model.save("rice_or_soup.model")

model.export_corem1("RiceSoupClassifier.mlmodel")
