import argparse
from model_utils import FlowerClassifier, save_checkpoint

def main():
    parser = argparse.ArgumentParser(description='Treinamento do Classificador de Flores')
    parser.add_argument('--lr', type=float, default=0.001, help='Taxa de aprendizado')
    parser.add_argument('--epochs', type=int, default=5, help='Número de épocas')
    parser.add_argument('--hidden_units', type=int, default=128, help='Número de unidades na camada oculta')
    parser.add_argument('--model_name', type=str, default='vgg16', help='Nome do modelo')
    args = parser.parse_args()

    classifier = FlowerClassifier(model_name=args.model_name, hidden_units=args.hidden_units, lr=args.lr, epochs=args.epochs)
    classifier.train()
    save_checkpoint(classifier.model, classifier.optimizer, classifier.epochs, classifier.train_loader.dataset.class_to_idx)

if __name__ == '__main__':
    main()
