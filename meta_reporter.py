"""
MetaReport CLASS
"""

import json
import requests


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args,
                                                                 **kwargs)
        return cls._instances[cls]


# MetaReporter CLASS definition
class MetaReporter(object, metaclass=Singleton):
    def __init__(self):
        self.initialized = False
        self.has_error = False

    def initialize(self, worker_id, logger=None):
        # self.url = 'http://metaai-stg-service.metaai-stg:5000/api/worker/{}/result/valid'.format(worker_id)  # debug msg on meta_learner web page
        self.url = 'http://stg.metaai.tbrain.com/api/worker/{}/result/valid'.format(worker_id)                # debug msg on training window
        self.header = {"Accept": "application/json",
                       "Accept-Encoding": "utf-8",
                       "Content-Type": "application/json"}

        self.print = logger.write if logger is not None else print
        self.initialized = True

    def _send(self, url, header, body):
        try:
            res = requests.put(url, headers=header, data=json.dumps(body))
            if res is None or res.status_code != 200:
                self.print(('Failed to send the request to Meta AI system ({})'
                           .format(url)))
                return

            self.print(('Successfully request to Meta AI system ({})'
                       .format(url)))
            self.print(body)
        except Exception as err:
            if not self.has_error:
                self.print(str(err))
                self.has_error = True

    def send_iteration(self, iteration, note_f1, note_overlap,
                       note_onset_offset_f1, note_f1_dataset2,
                       note_overlap_dataset2, note_onset_offset_f1_dataset2):
        def _make_dict(name, value, vtype='float'):
            return {
                       "iteration": iteration,
                       "name": name,
                       "valueType": vtype,
                       "value": str(value)
                   }

        assert self.initialized

        body = []
        body.append(_make_dict('note_f1', note_f1))
        body.append(_make_dict('note_overlap', note_overlap))
        body.append(_make_dict('note_onset_offset_f1', note_onset_offset_f1))
        body.append(_make_dict('note_f1_dataset2', note_f1_dataset2))
        body.append(_make_dict('note_overlap_dataset2', note_overlap_dataset2))
        body.append(_make_dict('note_onset_offset_f1_dataset2',
                    note_onset_offset_f1_dataset2))

        self._send(self.url, self.header, body)


if __name__ == "__main__":
    import torch
    # Meta Learner config parameters
    # (worker_id: any fixed integer number,
    #  iteration: the last iteration # before sending)
    worker_id = 12345
    iteration = 1

    # Evaluation metric sent from the Meta Learner
    note_f1 = 1
    note_overlap = 1
    note_onset_offset_f1 = 1
    note_f1_dataset2 = 1
    note_overlap_dataset2 = 1
    note_onset_offset_f1_dataset2 = 1

    # Configuration of the MetaReporter
    reporter = MetaReporter()
    reporter.initialize(worker_id)
    reporter.send_iteration(iteration, note_f1, note_overlap,
                            note_onset_offset_f1, note_f1_dataset2,
                            note_overlap_dataset2,
                            note_onset_offset_f1_dataset2)
