import pickle


def save_posterior(posterior, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(posterior, fp)


def load_posterior(filename):
    with open(filename, 'rb') as fp:
        posterior = pickle.load(fp)
    return posterior
