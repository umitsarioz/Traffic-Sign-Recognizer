from sklearn.model_selection import train_test_split

from preprocess import Preprocessor
from train import Trainer


def main(skip_preprocess=False, skip_train=False):
    processor_obj = Preprocessor(root_filepath='dataset', train_filename='Train.csv', test_filename='Test.csv')
    processor_obj.run(skip_preprocess=skip_preprocess)

    data = processor_obj.read_feature_and_labels_from_files()
    train_features, train_labels = data.get('train_features'), data.get('train_labels')
    train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels,
                                                                              test_size=.1,
                                                                              random_state=17, shuffle=True)

    # Trainer & Model
    train_obj = Trainer(lr=1e-4, n_epoch=5, width=32, height=32, n_label=train_labels.shape[1])
    if not skip_train:
        train_obj.train(X_train=train_features, y_train=train_labels, X_val=val_features, y_val=val_labels, save=True)
        train_obj.plot_acc_loss(save=True)
        model = train_obj.model
    else:
        model = train_obj.load_model()

    # Calculate Accuracy Scores & plot Them.
    test_features, test_labels = data.get('test_features'), data.get('test_labels')
    acc_test = train_obj.calculate_accuracy_score(model, x=test_features, y=test_labels)
    acc_train = train_obj.calculate_accuracy_score(model, x=train_features, y=train_labels)
    acc_val = train_obj.calculate_accuracy_score(model, x=val_features, y=val_labels)
    print(f"train-acc: %{acc_train:.3f} | val-acc: %{acc_val:.3f} | test-acc: %{acc_test:.3f}")


if __name__ == "__main__":
    main(skip_preprocess=True, skip_train=True)
