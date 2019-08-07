import json

food_prediction = {} ###{"text":"[false positive,false negative]"}
service_prediction = {}

def wrong_food_prediction_counter(file):
	for line in file:
		B = json.loads(line)
		if B['text'] not in food_prediction:
			food_prediction[B['text']] = [0,0]
			if B['predict label'] == 1:
				food_prediction[B['text']][0] += 1
			else:
				food_prediction[B['text']][1] += 1

		else:
			if B['predict label'] == 1:
				food_prediction[B['text']][0] += 1
			else:
				food_prediction[B['text']][1] += 1


def wrong_service_prediction_counter(file):
	for line in file:
		B = json.loads(line)
		if B['text'] not in service_prediction:
			service_prediction[B['text']] = [0,0]
			if B['predict label'] == 1:
				service_prediction[B['text']][0] += 1
			else:
				service_prediction[B['text']][1] += 1
		
		else:
			if B['predict label'] == 1:
				service_prediction[B['text']][0] += 1
			else:
				service_prediction[B['text']][1] += 1

f = open("lr_results/w2v_food_wrong_prediction.json", encoding='UTF-8')
wrong_food_prediction_counter(f)
f.close()

f = open("lr_results/w2v_service_wrong_prediction.json", encoding='UTF-8')
wrong_service_prediction_counter(f)
f.close()

f = open("lr_results/feature1_food_wrong_prediction.json", encoding='UTF-8')
wrong_food_prediction_counter(f)
f.close()

f = open("lr_results/feature1_service_wrong_prediction.json", encoding='UTF-8')
wrong_service_prediction_counter(f)
f.close()

f = open("lr_results/feature2_food_wrong_prediction.json", encoding='UTF-8')
wrong_food_prediction_counter(f)
f.close()

f = open("lr_results/feature2_service_wrong_prediction.json", encoding='UTF-8')
wrong_service_prediction_counter(f)
f.close()

f = open("lr_results/feature3_food_wrong_prediction.json", encoding='UTF-8')
wrong_food_prediction_counter(f)
f.close()

f = open("lr_results/feature3_service_wrong_prediction.json", encoding='UTF-8')
wrong_service_prediction_counter(f)
f.close()

f = open("lr_results/feature4_food_wrong_prediction.json", encoding='UTF-8')
wrong_food_prediction_counter(f)
f.close()

f = open("lr_results/feature4_service_wrong_prediction.json", encoding='UTF-8')
wrong_service_prediction_counter(f)
f.close()

f = open("lr_results/feature5_food_wrong_prediction.json", encoding='UTF-8')
wrong_food_prediction_counter(f)
f.close()

f = open("lr_results/feature5_service_wrong_prediction.json", encoding='UTF-8')
wrong_service_prediction_counter(f)
f.close()

f = open("lr_results/feature6_food_wrong_prediction.json", encoding='UTF-8')
wrong_food_prediction_counter(f)
f.close()

f = open("lr_results/feature6_service_wrong_prediction.json", encoding='UTF-8')
wrong_service_prediction_counter(f)
f.close()

f = open("lr_results/feature7_food_wrong_prediction.json", encoding='UTF-8')
wrong_food_prediction_counter(f)
f.close()

f = open("lr_results/feature7_service_wrong_prediction.json", encoding='UTF-8')
wrong_service_prediction_counter(f)
f.close()


f = open(r"lr_results/general_food_prediction.json", 'w', encoding='UTF-8')
for text,count in food_prediction.items():
	json.dump({"text": text, "false positive: ": count[0], "false negative: ": count[1]}, f)
	f.write('\n')

f = open(r"lr_results/general_service_prediction.json", 'w', encoding='UTF-8')
for text,count in service_prediction.items():
	json.dump({"text": text, "false positive: ": count[0], "false negative: ": count[1]}, f)
	f.write('\n')
