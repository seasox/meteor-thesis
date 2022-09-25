import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

from meteor_analysis import MeteorStatistic


def flat_map(f, xs):
    return [y for ys in xs for y in f(ys)]


def ld(fname):
    f = open(fname, "rb")
    import pickle
    o = pickle.load(f)
    f.close()
    return o


def create_plot(data, step_size, tikzexport):
    plt.figure(dpi=1200)
    mismatch_count = list(map(lambda x: len(x.mismatches), data))

    histbins = np.arange(min(mismatch_count) - 0.5, max(mismatch_count) + 1.5)
    weights = np.ones_like(mismatch_count) / len(mismatch_count)
    plt.xlim(min(histbins), max(histbins))
    plt.xticks(np.arange(min(mismatch_count), max(mismatch_count)+1, step=1))
    plt.xlabel('$x$')
    plt.ylabel('$\hat{Pr}[X=x]$')
    plt.hist(mismatch_count, bins=histbins, align='mid', weights=weights)
    plt.tight_layout()
    if tikzexport:
        axis_params = [
            'yticklabel style={/pgf/number format/fixed}',
        ]
        tikzplotlib.save(f'../tex/fig_meteor_stats_mismatch_count_{step_size}.tikz', extra_axis_parameters=axis_params)
    else:
        plt.show()
    plt.clf()


def visualize_mismatches(data):
    mismatches = list(flat_map(lambda x: x.mismatches, data))
    mismatches = [(len(tokenization[1]), '||'.join(tokenization[1]), '||'.join(tokenization[3])) for tokenization in mismatches]
    mismatches = '\\\\\n'.join([x[1] + ' & ' + x[2] for x in mismatches])
    return mismatches


def check_matching_tokenizations(data: [MeteorStatistic]):
    def pred(x):
        return (len(x.mismatches) == 0 and x.encoded_tokens != x.decoded_tokens) \
            or (len(x.mismatches) != 0 and x.encoded_tokens == x.decoded_tokens)
    return list(filter(pred, data))

if __name__ == '__main__':
    tikzexport = True
    for step_size in [128, 1024]:
        print(f'generating {step_size} bytes stats')
        fname = f'meteor_statistics_{step_size}.pickle'
        data = ld(fname)
        data_w_coding = list(filter(lambda x: x.coding == 'arithmetic', data))
        if not data_w_coding:
            raise 'no data'
        no_mismatch_prob=len(list(filter(lambda x: len(x.mismatches) == 0, data_w_coding)))/len(data_w_coding)
        print(len(data_w_coding))
        print(f'Pr_{step_size}[X=0]={no_mismatch_prob}')
        # create_plot(data_w_coding, step_size, tikzexport)
        # just a little test if the data makes sense
        print(check_matching_tokenizations(data_w_coding))
