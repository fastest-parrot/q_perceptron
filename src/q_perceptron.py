from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
from IPython.display import Markdown as md
from IPython.display import display
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, mean_absolute_error


class QPercetron:
    def __init__(self):
        self.random_state = 5

    def __sigmoid__(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_prime__(self, x):
        return self.__sigmoid__(x) * (1 - self.__sigmoid__(x))

    def activation(self, x):
        return self.__sigmoid__(x)

    def activation_prime(self, x):
        return self.__sigmoid_prime__(x)

    def score_model(self, predictions, target):
        return {
            'accuracy': accuracy_score(np.real(predictions), np.real(target)),
            'mae': mean_absolute_error(np.real(predictions), np.real(target))
        }

    def fit(self,
            features,
            target,
            features_test=None,
            target_test=None,
            random_state=random_state,
            bias=0.5,
            learning_rate=0.10,
            iterations=10000,
            verbose=False,
            feature_names=[],
            brake_callback=None,
            plot=False):

        np.random.seed(self.__random_state__)
        weights = np.array([[x]
                            for x in np.random.random((features.dims[1][0]))])
        results = []
        for epoch in range(iterations):
            inputs = Qobj(features)
            in_o = inputs * Qobj(weights) + bias
            out_o = Qobj(self.activation(in_o))
            loss = out_o - Qobj(target)
            if verbose:
                print(loss)

            derror_douto = loss
            douto_dino = Qobj(self.activation_prime(out_o))
            deriv = derror_douto
            for r, c in enumerate(deriv):  #does this do a hadamard product?
                deriv.data[r] *= douto_dino.data[r]

            inputs = inputs.trans()
            deriv_final = inputs * deriv
            weights -= learning_rate * deriv_final

            for i in deriv:
                bias -= learning_rate * i

            epoch_results = {
                f'weight_{i}': weight[0][0]
                for i, weight in enumerate(weights)
            }

            epoch_results.update({
                'epoch': epoch,
                'bias': bias[0][0],
            })
            predict_test = self.predict(epoch_results, features_test)
            test_scoring = self.score_model(predict_test, target_test)
            epoch_results.update(test_scoring)
            print(
                f'Epoch:{epoch} Test Acc: {test_scoring["accuracy"]:.6f} Test MAE: {test_scoring["mae"]:.6f}',
                end='\r')

            #check for brake callback
            if epoch >= brake_callback.patience:
                all_scores = {
                    'accuracy': [x['accuracy'] for x in results],
                    'mae': [x['mae'] for x in results]
                }  #TODO: throw these in a separate dict to reduce overhead, also make more abstract instead of hc scores
                if not brake_callback.should_continue(all_scores):
                    display(md(f'***Early Stoppage***'))
                    display(md(f'- Epoch: *{epoch}*'))
                    display(
                        md(f'- Last Value: *{brake_callback.last_val:.6f}*'))
                    display(
                        md(f'- Test Value: *{brake_callback.compare_to:.6f}*'))
                    display(md(f'- Function: *{brake_callback.func}*'))
                    display(md(f'- Score: *{brake_callback.stat}*'))
                    display(
                        md(f'- Criterion: *{brake_callback.compare_type}*'))
                    display(
                        md(f'- Tolerance: *{brake_callback.tolerance:.6f}*'))
                    epoch_results.update({'brake_callback': brake_callback})
                    break
                else:
                    epoch_results.update({'brake_callback': None})

            results.append(epoch_results)
            if plot:
                self.plot(results, feature_names)

        return results

    def predict(self, model, test, cutoff=0.5):
        weights = Qobj(np.array([y for x, y in model.items()
                                 if 'weight' in x]))
        bias = model['bias']
        return Qobj(
            [[self.softmax(activation(test[i][0].T * weights + bias), cutoff)]
             for i in range(0, test.shape[0])])

    def softmax(self, val, cutoff):
        return 1 if val >= cutoff else 0

    def plot(self, nnet_results, feature_names, score='accuracy'):
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 20))
        weight_plots = []
        for i, feature in enumerate(feature_names):
            p = sns.lineplot(
                x=[x['epoch'] for x in nnet_results],
                y=[np.real(x[f'weight_{i}']) for x in nnet_results],
                ax=ax[0],
                label=f'{feature_names[i]}')
            weight_plots.append(p)
        p3 = sns.lineplot(x=[x['epoch'] for x in nnet_results],
                          y=[np.real(x['bias']) for x in nnet_results],
                          c='green',
                          ax=ax[1])
        p4 = sns.lineplot(x=[x['epoch'] for x in nnet_results][1:],
                          y=[np.real(x[score]) for x in nnet_results][1:],
                          c='orange',
                          ax=ax[2])
        weight_plots.append(p3)
        weight_plots.append(p4)
        for f in weight_plots:
            f.set_xlabel('Epoch')
            f.set_xticklabels([f'{x:,.0f}' for x in f.get_xticks()])
            f.set_yticklabels([f'{y:,.3f}' for y in f.get_yticks()])

        p3.set_title('Bias by Epoch')
        p4.set_title(f'{score} by Epoch')


class BrakeCondition:
    def __init__(self,
                 stat='accuracy',
                 patience=5,
                 tolerance=0.005,
                 compare_type='better',
                 func=None):
        self.stat = stat  #'accuracy or mae'
        self.patience = patience
        self.func = func  #a func to be applied to brake - np.mean, np.median etc?
        self.tolerance = tolerance
        self.compare_type = compare_type

    #TODO: add more rolling-type brakes?
    def should_continue(self, scores):
        check = scores[self.stat]
        self.last_val = check[-1]
        #compare_to = check[-self.patience]
        if self.func:
            self.compare_to = self.func(
                check[-self.patience:-2]
            )  #compare to function inclusive or exclusive of last element?
        else:  #compare to raw value some steps in past
            self.compare_to = check[-self.patience]
        if self.compare_type == 'better':
            return False if (
                self.last_val - self.compare_to
            ) < self.tolerance else True  #stats are getting higher/better (Accuracy, R-squared etc)
        else:
            return False if (
                self.last_val - self.compare_to
            ) > self.tolerance else True  #stats are getting lower/better (RMSE, MAE)


#TODO: wrap these into the class after setting a method for storing results of .fit()


def weight_report(models, feature_names=[]):
    report = []
    for j, feature in enumerate(feature_names):
        #print(feature)
        all_weights = []
        for i, (random_state, model) in enumerate(models.items()):
            #gather all the values this weight across models/epochs
            #model_weights = [y for x, y in model.items() if f'weight_{j}' in x]
            model_weights = []
            for m in model:
                #get each value by epoch
                [
                    model_weights.append(np.real(y + m['bias']))
                    for x, y in m.items() if x == f'weight_{j}'
                ]
            report.append({
                'feature': feature,
                'random_state': random_state,
                'mean': np.mean(model_weights),
                'median': np.median(model_weights),
                'max': np.max(model_weights),
                'min': np.min(model_weights),
                'std': np.std(model_weights)
            })
            #print(random_state, len(model_weights))
        #print(report)
        #break
        #all_weights.extend(model_weights)
    report = pd.DataFrame(report).set_index(['feature'])

    fig, ax = plt.subplots(nrows=len(feature_names),
                           ncols=1,
                           figsize=(20, 120),
                           sharex=True)
    ax.ravel()
    for i, (feature, data) in enumerate(report.groupby(lambda x: x)):
        #print(feature)
        sns.scatterplot(x=data['random_state'],
                        y=data['mean'],
                        ax=ax[i],
                        label='mean')
        sns.scatterplot(x=data['random_state'],
                        y=data['max'],
                        ax=ax[i],
                        label='max')
        sns.scatterplot(x=data['random_state'],
                        y=data['min'],
                        ax=ax[i],
                        label='min')
        sns.scatterplot(x=data['random_state'],
                        y=data['median'],
                        ax=ax[i],
                        label='median')
        sns.lineplot(x=data['random_state'],
                     y=[np.mean(data['mean'])] * len(data['random_state']),
                     ax=ax[i],
                     label='total_avg')

        ax[i].set_ylabel(f'${feature.upper()}$')
        ax[i].lines[0].set_linestyle('dotted')

    plt.tight_layout()

    return report


def train_test_split_nn(self, features, target):
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        stratify=[np.real(x)
                  for x in target.data],  #make sure we stratify properly
        random_state=random_state,
        train_size=0.8,
        test_size=0.2)
    return Qobj(X_train), Qobj(X_test), Qobj(y_train), Qobj(y_test)
