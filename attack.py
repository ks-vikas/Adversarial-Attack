import torch
import matplotlib.pyplot as plt
import numpy as np
from model import ANNClassifier

class FGSM:
    def __init__(self, model, criterion, epsilon=0.3):
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon

    def apply(self, test_loader):
        adversarial_examples = []
        misclassified_count = 0

        for images, labels in test_loader:
            images.requires_grad = True
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # FGSM attack
            perturbed_images = images + self.epsilon * torch.sign(images.grad)
            perturbed_images = torch.clamp(perturbed_images, 0, 1)

            # Check misclassification
            perturbed_outputs = self.model(perturbed_images)
            perturbed_labels = perturbed_outputs.argmax(dim=1)
            misclassified_count += (perturbed_labels != labels).sum().item()

            adversarial_examples.append(perturbed_images)

        evasion_rate = misclassified_count / len(test_loader.dataset)
        return {"evasion_rate": evasion_rate, "adv_examples": adversarial_examples}

if __name__ == "__main__":
    # Load the trained model
    model = ANNClassifier(input_size=28*28, hidden_size=128, output_size=10, train_params={"learning_rate": 0.001, "epochs": 10})
    model.load_state_dict(torch.load("model.pth"))

    # Load the test data
    _,_, test_loader = model.get_dataloaders(batch_size=1)

    # Define the criterion (loss function)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Apply FGSM attack
    attack = FGSM(model, criterion, epsilon=0.3)
    results = attack.apply(test_loader)

    print("Evasion Rate:", results["evasion_rate"]*100)

########################################### Plot graph for multiple epsilon #################################################
    # # Apply FGSM attack
    # eps = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    # evasion_rate = []
    # for e in eps:
    #     attack = FGSM(model, criterion, epsilon=e)
    #     results = attack.apply(test_loader)
    #     evasion_rate.append(results["evasion_rate"]*100)
    #     print("Evasion Rate:", results["evasion_rate"]*100)

    # # Plotting the graph
    # plt.plot(eps, evasion_rate, '-o')  # 'o' for markers (you can change marker style)

    # # Adding labels and title
    # plt.xlabel('Epsilon Value')
    # plt.ylabel('Evasion Rate')
    # plt.title('Epsilon vs Evasion Rate graph')

    # # Display the plot
    # # plt.grid(True)  # Show grid
    # plt.show()
############################################################################################

    # Question 2: Plot Original Images, Adversarial Examples, and Noise
    # Assuming test_loader is defined and contains the test dataset
    random_images = []
    temp = results["adv_examples"]
    adversarial_images = []
    # Select one image for each digit
    for digit in range(10):
        digit_indices = (test_loader.dataset.targets == digit).nonzero(as_tuple=True)[0]
        image_index = digit_indices[0]  # Select the first image for each digit
        random_image, label = test_loader.dataset[image_index]
        random_images.append(random_image)
        adversarial_images.append(temp[image_index])

    # Plotting the images
    num_samples = 10
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 2, 6))

    for i, (image, adv_image) in enumerate(zip(random_images, adversarial_images)):
        # Original Image
        axes[0, i].imshow(image.squeeze(), cmap='gray')
        axes[0, i].set_title(f'Original {i}')
        axes[0, i].axis('off')

        # Adversarial Example
        axes[1, i].imshow(adv_image.squeeze().detach().numpy(), cmap='gray')
        axes[1, i].set_title(f'Adversarial {i}')
        axes[1, i].axis('off')

        # Adversarial Noise (L2 Norm)
        noise = (adv_image - image).detach().numpy()
        l2_norm = np.linalg.norm(noise)
        axes[2, i].imshow(noise.squeeze(), cmap='gray')
        axes[2, i].set_title(f'L2 Norm: {l2_norm:.2f}')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()


    # Question 3: Identify Most Misclassified Digit
    # Randomly pick one digit
    digit = np.random.randint(0, 10)

    misclassified_counts = []

    for i in range(10):
        misclassified_count = 0
        for images, labels in test_loader:
            if labels.item() == digit:
                with torch.enable_grad():
                    images.requires_grad = True  # Ensure requires_grad is set to True
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    perturbed_images = images + 0.1 * torch.sign(images.grad)
                    perturbed_images = torch.clamp(perturbed_images, 0, 1)
                    perturbed_outputs = model(perturbed_images)
                    perturbed_labels = perturbed_outputs.argmax(dim=1)
                    if perturbed_labels.item() != digit:
                        misclassified_count += 1
        misclassified_counts.append(misclassified_count)


    most_misclassified_digit = np.argmax(misclassified_counts)

    print(f"The digit {digit} got misclassified the most to class {most_misclassified_digit}.")