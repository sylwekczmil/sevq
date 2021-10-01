from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

from research.helpers.util import seed_everything

pandas2ri.activate()
import pandas as pd

from research.helpers.data import RESEARCH_DIR


def process_scott_knott(normalized=True):
    name = 'comparison_normalized' if normalized else 'comparison'
    print(f'Processing {name} scott_knott')
    sk = importr('ScottKnottESD')
    grdevices = importr('grDevices')
    plot = robjects.r('plot')

    scott_knott_dir = RESEARCH_DIR.joinpath('scott_knott')
    scott_knott_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESEARCH_DIR.joinpath(f'{name}.csv'))
    algorithms = df['algorithm'].unique()
    for metric in ['accuracy', 'auc']:
        data = {}
        for a in algorithms:
            a_data = df.loc[df['algorithm'] == a]
            data[a] = a_data[metric].values

        data = pd.DataFrame(data=data)
        data.to_csv(scott_knott_dir.joinpath(f'{name}_{metric}.csv'), index=False)

        r_sk = sk.sk_esd(data)
        y_label = 'AUC' if metric == 'auc' else 'Accuracy'
        print(y_label, r_sk)

        ps_im_path = str(scott_knott_dir.joinpath(f'{name}_{metric}.eps'))
        grdevices.postscript(file=ps_im_path, horizontal=False, width=11, height=8.5, paper='letter')
        plot(r_sk, main="", xlab="", ylab=y_label, lwd=4, las=2)
        grdevices.dev_off()

        with open(ps_im_path, 'r') as f:
            data = f.readlines()
        del data[-5]
        del data[-5]
        data = ''.join(data)
        with open(ps_im_path, 'w') as f:
            f.write(data)

        png_im_path = str(scott_knott_dir.joinpath(f'{name}_{metric}.png'))
        grdevices.png(file=png_im_path)
        plot(r_sk, main="", xlab="", ylab=y_label, lwd=4, las=2)
        grdevices.dev_off()


if __name__ == '__main__':
    seed_everything()
    for normalized in [False, True]:
        try:
            process_scott_knott(normalized=normalized)
        except:
            pass
