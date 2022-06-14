
md = {}

for image,label, xxx_, filenam in dataloder:
	outpus  = model(xx)
	pred = outpus.cpu().detach().numpy().tolist()
	for idx in range(image.shape[0]):
		md[filenam[idx]] = pred

pickle.dump(md)
