import os
import pandas as pd


class PandasWriter:
    def __init__(self, log_dir, num_epoch):
        self.log_dir = log_dir
        self.log_path = os.path.join(self.log_dir, "log.csv")
        self.num_epoch = num_epoch
        self._initialize()

    def _initialize(self):
        if os.path.exists(self.log_path):
            # 過去のログ読み込み
            self.log_df = pd.read_csv(self.log_path)
        else:
            # 新規ログファイル作成
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_df = pd.DataFrame({'epoch': range(1, self.num_epoch+1)})
            self.log_df.to_csv(self.log_path, index=False)

    def add_scalar(self, tag, scalar_value, epoch):
        self._initialize()
        self.log_df.loc[self.log_df['epoch'] == epoch, tag] = scalar_value
        self.log_df.to_csv(self.log_path, index=False)

    def close(self):
        pass
