# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
The aim of this project is to develop a Convolutional Neural Network (CNN) model capable of classifying fashion product images from the Fashion MNIST dataset. The dataset contains grayscale images representing various clothing items and accessories, including shirts, shoes, bags, trousers, and dresses. The model is designed to identify and assign the correct fashion category to each image with high accuracy and reliability.

Training dataset: 60,000 fashion images
Testing dataset: 10,000 fashion images
Output classes: 10 different fashion item categories

The proposed CNN architecture uses several convolutional layers to extract image features, activation functions to introduce non-linearity, and pooling layers to reduce dimensionality. Finally, fully connected dense layers are used to perform classification and generate predictions for all ten fashion categories.

## Neural Network Model
<img width="372" height="243" alt="image" src="https://github.com/user-attachments/assets/9d312473-089e-4edf-893f-5003fa6c274e" />


## DESIGN STEPS
## STEP 1:

Import the necessary libraries such as PyTorch, Torchvision, NumPy, and Matplotlib. Load the Fashion MNIST dataset and apply preprocessing techniques like normalization and tensor conversion.

## STEP 2:

Separate the dataset into training and testing datasets. Create DataLoader objects to process the images in batches during model training and testing.

## STEP 3:

Design the Convolutional Neural Network (CNN) architecture using convolution layers, activation functions like ReLU, pooling layers, and fully connected dense layers for image classification.

## STEP 4:

Initialize the CNN model and define the loss function using CrossEntropyLoss. Select the Adam optimizer to update the model weights during training.

## STEP 5:

Train the CNN model by performing forward propagation, calculating loss, backpropagation, and updating weights for several training epochs.

## STEP 6:

Test and evaluate the trained CNN model using the test dataset. Measure the prediction accuracy and classify unseen fashion images into the correct categories.






## PROGRAM

### Name:NISHA J

### Register Number:212223040133

```
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # write your code here
        self.conv1= nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2= nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3= nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)
    def forward(self, x):
        # write your code here
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x



# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

#Train the Model
def train_model(model, train_loader, num_epochs=3):

    # write your code here
    for epoch in range(num_epochs):
      model.train()
      running_loss=0.0
      for images, labels in train_loader:
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()


        print('Name: Nisha J')
        print('Register Number:212223230043')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')


```

### OUTPUT

## Training Loss per Epoch
<img width="372" height="296" alt="image" src="https://github.com/user-attachments/assets/80720ec5-84f7-4ff9-bfcc-f07a920073d5" />


## Confusion Matrix
<img width="883" height="733" alt="image" src="https://github.com/user-attachments/assets/e8893ad1-b575-4bc8-8289-e66f42df5d71" />


## Classification Report
<img width="600" height="418" alt="image" src="https://github.com/user-attachments/assets/08dc550c-810e-4675-ac6d-fe6f8d1c93c2" />


### New Sample Data Prediction
<img width="561" height="561" alt="image" src="https://github.com/user-attachments/assets/aae9a94a-6098-4d4a-952c-a531999642e3" />


## RESULT
Thus,the Convolutional Neural Network (CNN) model was successfully trained and achieved good classification performance on the given image dataset.
