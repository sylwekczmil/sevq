import io
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import List
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn import preprocessing

DATA_URL = 'https://sci2s.ugr.es/keel/dataset/data/classification//full/All.zip'
DATA_DIR = Path(__file__).parent.parent.parent.joinpath('data')
CLASSIFICATION_DATA_DIR = DATA_DIR.joinpath('classification')
RESEARCH_DIR = DATA_DIR.joinpath('research')


def download_data():
    if not CLASSIFICATION_DATA_DIR.exists():
        print('Downloading the data, may take a while...')
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        resp = urlopen(DATA_URL)
        zipfile = ZipFile(BytesIO(resp.read()))
        for file in zipfile.namelist():
            if file.startswith('classification/'):
                zipfile.extract(file, DATA_DIR)
        print('Download complete')


@dataclass
class FoldData:
    index: int
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    labels: np.ndarray

    def normalize(self):
        x_tra_len = len(self.x_train)
        x = np.concatenate([self.x_train.astype(float), self.x_test.astype(float)])
        min_max_scaler = preprocessing.MinMaxScaler()
        x = min_max_scaler.fit_transform(x)
        self.x_train, self.x_test = x[:x_tra_len], x[x_tra_len:]


@dataclass
class Dataset:
    name: str
    path: Path
    header: str = field(default='')
    types: List[str] = field(default=lambda: [])

    def data(self):
        data_path = self.path.joinpath(f'{self.name}.dat')
        x, y = self._load_data(data_path)
        return x, y

    def df(self):
        data_path = self.path.joinpath(f'{self.name}.dat')
        return self._load_data(data_path, return_df=True)

    def labels(self):
        _, y = self.data()
        return np.unique(y)

    def data_10_fold(self):
        labels = self.labels()
        for fold in range(1, 11):
            x_tra, y_tra = self._load_data(self.path.joinpath(f'{self.name}-10-{fold}tra.dat'))
            x_tst, y_tst = self._load_data(self.path.joinpath(f'{self.name}-10-{fold}tst.dat'))
            yield FoldData(
                index=fold,
                x_train=x_tra,
                y_train=y_tra,
                x_test=x_tst,
                y_test=y_tst,
                labels=labels
            )

    def data_5_fold(self):
        labels = self.labels()
        for fold in range(1, 6):
            x_tra, y_tra = self._load_data(self.path.joinpath(f'{self.name}-5-{fold}tra.dat'))
            x_tst, y_tst = self._load_data(self.path.joinpath(f'{self.name}-5-{fold}tst.dat'))
            yield FoldData(
                index=fold,
                x_train=x_tra,
                y_train=y_tra,
                x_test=x_tst,
                y_test=y_tst,
                labels=labels
            )

    def __post_init__(self):
        self._load_header()
        self._load_types()

    def _load_types(self):
        df = self._load_data(self.path.joinpath(f'{self.name}-10-1tst.dat'), return_df=True)
        self.types = [str(d) for d in df.dtypes]

    def _load_header(self):
        header_path = self.path.joinpath(f'{self.name}.dat')
        attributes = []
        attribute_types = []
        inputs = []
        outputs = []
        for line in header_path.open('r'):
            if '@attribute' in line:
                if '{' in line:
                    name = line.split('{')[0].split()[1]
                    type = '-'
                else:
                    s = line.split()[1:]
                    name = s[0].strip()
                    type = s[1].split('[')[0].strip()
                attributes.append(name)
                attribute_types.append(type)
            if '@input' in line:
                inputs.append(line.split()[1:])
            elif '@output' in line:
                outputs.append(line.split()[1])
            elif '@data' in line:
                break
        if len(outputs) < 1:
            outputs = ['Class']

        class_name = outputs[0]
        self.header = ','.join(attributes).replace(class_name, 'Class').replace(' ', '')
        if self.header == '':
            raise Exception('Malformed data file')

    def _load_data(self, dat_path: Path, return_df=False):
        csv_lines = [self.header]
        found_data = False
        for line in dat_path.open('r'):
            if found_data:
                csv_lines.append(line)
            else:
                found_data = '@data' in line
        csv = io.StringIO('\n'.join(csv_lines))
        df = pd.read_csv(csv)
        y = df['Class'].astype('category').cat.codes.values
        if return_df:
            return df
        del df['Class']
        x = df.values
        return x, y


def datasets(only_numerical=False):
    for path in CLASSIFICATION_DATA_DIR.glob('*'):
        try:
            ds = Dataset(
                name=path.name,
                path=path
            )
            if only_numerical and 'object' in ds.types:
                continue
            yield ds
        except Exception as e:
            pass
