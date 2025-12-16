import pandas as pd
from dataset import Dataset
from model import Model

def main():
    data = pd.read_csv('spam.csv', encoding='latin-1')
    
    columns = data.columns
    data = data[[columns[0], columns[1]]]
    data.columns = ['label', 'message']
    
    dataset = Dataset(data['message'].tolist(), data['label'].tolist())
    
    dataset.split_dataset(val=0.15, test=0.15)
    
    model = Model(alpha=1)
    model.fit(dataset)
    
   
    val_acc = model.validation()
    test_acc = model.test()
    
    print(f"Validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    
    test_messages = [
        "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",
        "I HAVE A DATE ON SUNDAY WITH WILL!!",
        "FREE entry into our £100 weekly draw. Just text WIN to 80088 now!",
        "Hey, don't forget to bring the documents for the meeting today."
    ]
    
    for msg in test_messages:
        prediction = model.inference(msg)
        print(f"Message: {msg[:50]}... -> Prediction: {prediction}")
    
    
    if val_acc >= 0.95 and test_acc >= 0.95:
        print("\nAccuracy > 95% on both validation and test sets!")
    else:
        print("\nTry changing the model settings.")

if __name__ == "__main__":
    main()