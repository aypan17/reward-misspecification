"""
Based on rllab's logger.

https://github.com/rll/rllab
"""
import csv
import datetime
import errno
import json
import os
import os.path as osp
import joblib
import pickle
import sys
from contextlib import contextmanager
from enum import Enum

import dateutil.tz
import numpy as np
import torch

from bgp.rlkit.core.tabulate import tabulate


class TerminalTablePrinter(object):
    def __init__(self):
        self.headers = None
        self.tabulars = []

    def print_tabular(self, new_tabular):
        if self.headers is None:
            self.headers = [x[0] for x in new_tabular]
        else:
            assert len(self.headers) == len(new_tabular)
        self.tabulars.append([x[1] for x in new_tabular])
        self.refresh()

    def refresh(self):
        import os
        rows, columns = os.popen('stty size', 'r').read().split()
        tabulars = self.tabulars[-(int(rows) - 3):]
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(tabulate(tabulars, self.headers))
        sys.stdout.write("\n")


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class Logger(object):
    def __init__(self):
        self._prefixes = []
        self._prefix_str = ''

        self._tabular_prefixes = []
        self._tabular_prefix_str = ''

        self._tabular = []

        self._text_outputs = []
        self._tabular_outputs = []

        self._text_fds = {}
        self._tabular_fds = {}
        self._tabular_header_written = set()

        self._snapshot_dir = None
        self._snapshot_mode = 'all'
        self._snapshot_gap = 1

        self._log_tabular_only = False
        self._header_printed = False
        self.table_printer = TerminalTablePrinter()

    def reset(self):
        self.__init__()

    def _add_output(self, file_name, arr, fds, mode='a'):
        if file_name not in arr:
            mkdir_p(os.path.dirname(file_name))
            arr.append(file_name)
            fds[file_name] = open(file_name, mode)

    def _remove_output(self, file_name, arr, fds):
        if file_name in arr:
            fds[file_name].close()
            del fds[file_name]
            arr.remove(file_name)

    def push_prefix(self, prefix):
        self._prefixes.append(prefix)
        self._prefix_str = ''.join(self._prefixes)

    def add_text_output(self, file_name):
        self._add_output(file_name, self._text_outputs, self._text_fds,
                         mode='a')

    def remove_text_output(self, file_name):
        self._remove_output(file_name, self._text_outputs, self._text_fds)

    def add_tabular_output(self, file_name, relative_to_snapshot_dir=False):
        if relative_to_snapshot_dir:
            file_name = osp.join(self._snapshot_dir, file_name)
        self._add_output(file_name, self._tabular_outputs,
                         self._tabular_fds,
                         mode='w')

    def remove_tabular_output(self, file_name,
                              relative_to_snapshot_dir=False):
        if relative_to_snapshot_dir:
            file_name = osp.join(self._snapshot_dir, file_name)
        if self._tabular_fds[file_name] in self._tabular_header_written:
            self._tabular_header_written.remove(
                self._tabular_fds[file_name])
        self._remove_output(file_name, self._tabular_outputs,
                            self._tabular_fds)

    def set_snapshot_dir(self, dir_name):
        self._snapshot_dir = dir_name

    def get_snapshot_dir(self, ):
        return self._snapshot_dir

    def get_snapshot_mode(self, ):
        return self._snapshot_mode

    def set_snapshot_mode(self, mode):
        self._snapshot_mode = mode

    def get_snapshot_gap(self, ):
        return self._snapshot_gap

    def set_snapshot_gap(self, gap):
        self._snapshot_gap = gap

    def set_log_tabular_only(self, log_tabular_only):
        self._log_tabular_only = log_tabular_only

    def get_log_tabular_only(self, ):
        return self._log_tabular_only

    def log(self, s, with_prefix=True, with_timestamp=True):
        out = s
        if with_prefix:
            out = self._prefix_str + out
        if with_timestamp:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
            out = "%s | %s" % (timestamp, out)
        if not self._log_tabular_only:
            # Also log to stdout
            print(out)
            for fd in list(self._text_fds.values()):
                fd.write(out + '\n')
                fd.flush()
            sys.stdout.flush()

    def record_tabular(self, key, val):
        self._tabular.append(
            (self._tabular_prefix_str + str(key), str(val)))

    def push_tabular_prefix(self, key):
        self._tabular_prefixes.append(key)
        self._tabular_prefix_str = ''.join(self._tabular_prefixes)

    def pop_tabular_prefix(self, ):
        del self._tabular_prefixes[-1]
        self._tabular_prefix_str = ''.join(self._tabular_prefixes)

    def save_extra_data(self, data, file_name='extra_data.pkl',
                        mode='joblib'):
        """
        Data saved here will always override the last entry

        :param data: Something pickle'able.
        """
        file_name = osp.join(self._snapshot_dir, file_name)
        if mode == 'joblib':
            import joblib
            joblib.dump(data, file_name, compress=3)
        elif mode == 'pickle':
            pickle.dump(data, open(file_name, "wb"))
        else:
            raise ValueError("Invalid mode: {}".format(mode))
        return file_name

    def get_table_dict(self, ):
        return dict(self._tabular)

    def get_table_key_set(self, ):
        return set(key for key, value in self._tabular)

    @contextmanager
    def prefix(self, key):
        self.push_prefix(key)
        try:
            yield
        finally:
            self.pop_prefix()

    @contextmanager
    def tabular_prefix(self, key):
        self.push_tabular_prefix(key)
        yield
        self.pop_tabular_prefix()

    def log_variant(self, log_file, variant_data):
        mkdir_p(os.path.dirname(log_file))
        with open(log_file, "w") as f:
            json.dump(variant_data, f, indent=2, sort_keys=True,
                      cls=MyEncoder)

    def record_tabular_misc_stat(self, key, values, placement='back'):
        if placement == 'front':
            prefix = ""
            suffix = key
        else:
            prefix = key
            suffix = ""
        if len(values) > 0:
            self.record_tabular(prefix + "Average" + suffix,
                                np.average(values))
            self.record_tabular(prefix + "Std" + suffix, np.std(values))
            self.record_tabular(prefix + "Median" + suffix,
                                np.median(values))
            self.record_tabular(prefix + "Min" + suffix, np.min(values))
            self.record_tabular(prefix + "Max" + suffix, np.max(values))
        else:
            self.record_tabular(prefix + "Average" + suffix, np.nan)
            self.record_tabular(prefix + "Std" + suffix, np.nan)
            self.record_tabular(prefix + "Median" + suffix, np.nan)
            self.record_tabular(prefix + "Min" + suffix, np.nan)
            self.record_tabular(prefix + "Max" + suffix, np.nan)

    def dump_tabular(self, *args, **kwargs):
        wh = kwargs.pop("write_header", None)
        if len(self._tabular) > 0:
            if self._log_tabular_only:
                self.table_printer.print_tabular(self._tabular)
            else:
                for line in tabulate(self._tabular).split('\n'):
                    self.log(line, *args, **kwargs)
            tabular_dict = dict(self._tabular)
            # Also write to the csv files
            # This assumes that the keys in each iteration won't change!
            for tabular_fd in list(self._tabular_fds.values()):
                writer = csv.DictWriter(tabular_fd,
                                        fieldnames=list(
                                            tabular_dict.keys()))
                if wh or (
                        wh is None and tabular_fd not in self._tabular_header_written):
                    writer.writeheader()
                    self._tabular_header_written.add(tabular_fd)
                writer.writerow(tabular_dict)
                tabular_fd.flush()
            del self._tabular[:]

    def pop_prefix(self, ):
        del self._prefixes[-1]
        self._prefix_str = ''.join(self._prefixes)

    def save_itr_params(self, itr, params, best=False):
        if best:            
            policy_file_name = osp.join(self._snapshot_dir, 'policy_best.pt')
            qf_file_name = osp.join(self._snapshot_dir, 'qf_best.pt')
            vf_file_name = osp.join(self._snapshot_dir, 'vf_best.pt')
            target_vf_file_name = osp.join(self._snapshot_dir, 'target_vf_best.pt')
            state_dict_file_name = osp.join(self._snapshot_dir, 'state_dict_best.pt')

            policy_cache = params['policy'].cpu()
            qf_cache = params['qf'].cpu()
            vf_cache = params['vf'].cpu()
            target_vf_cache = params['target_vf'].cpu()

            torch.save(policy_cache, policy_file_name)
            torch.save(qf_cache, qf_file_name)
            torch.save(vf_cache, vf_file_name)
            torch.save(target_vf_cache, target_vf_file_name)
            torch.save({'policy': policy_cache.state_dict(),
                        'qf': qf_cache.state_dict(),
                        'vf': vf_cache.state_dict(),
                        'target_vf_cache': target_vf_cache.state_dict()}, state_dict_file_name)

            mdl = params['eval_policy']
            orig_run_device = mdl.stochastic_policy.device
            params['policy'].to(orig_run_device)
            params['qf'].to(orig_run_device)
            params['vf'].to(orig_run_device)
            params['target_vf'].to(orig_run_device)
            params['alpha'].to(orig_run_device)

        if self._snapshot_dir:
            if self._snapshot_mode == 'all':
                file_name = osp.join(self._snapshot_dir, 'itr_%d.pkl' % itr)
                joblib.dump(params, file_name)
                eval_file_name = osp.join(self._snapshot_dir, 'eval_itr_%d.pkl' % itr)
                joblib.dump(params['eval_policy'], eval_file_name)
            elif self._snapshot_mode == 'last':
                # override previous params
                file_name = osp.join(self._snapshot_dir, 'params.pkl')
                joblib.dump(params, file_name)
                eval_file_name = osp.join(self._snapshot_dir, 'eval_params.pkl')
                joblib.dump(params['eval_policy'], eval_file_name)
            elif self._snapshot_mode == "gap":
                if itr % self._snapshot_gap == 0:
                    file_name = osp.join(self._snapshot_dir,
                                         'itr_%d.pkl' % itr)
                    joblib.dump(params, file_name)
                    eval_file_name = osp.join(self._snapshot_dir, 'eval_itr_%d.pkl' % itr)
                    joblib.dump(params['eval_policy'], eval_file_name)
            elif self._snapshot_mode == "gap_and_last":
                if itr % self._snapshot_gap == 0:
                    #file_name = osp.join(self._snapshot_dir,
                    #                     'itr_%d.pkl' % itr)
                    #joblib.dump(params, file_name)
                    #print('saving')
                    eval_file_name = osp.join(self._snapshot_dir, 'eval_itr_%d.pkl' % itr)
                    mdl = params['eval_policy']
                    orig_run_device = mdl.stochastic_policy.device
                    run_device = 'cpu'
                    mdl.stochastic_policy.device = run_device
                    mdl.stochastic_policy = mdl.stochastic_policy.to(run_device)
                    if mdl.stochastic_policy.features is not None:
                        mdl.stochastic_policy.features.device = run_device
                        mdl.stochastic_policy.features = mdl.stochastic_policy.features.to(run_device)
                    joblib.dump(mdl, eval_file_name)
                    mdl.stochastic_policy.device = orig_run_device
                    mdl.stochastic_policy = mdl.stochastic_policy.to(orig_run_device)
                    if mdl.stochastic_policy.features is not None:
                        mdl.stochastic_policy.features.device = orig_run_device
                        mdl.stochastic_policy.features = mdl.stochastic_policy.features.to(orig_run_device)
                    cache_file_name = osp.join(self._snapshot_dir, 'cache_itr_%d.pkl' % itr)
                    qf_cache = params['qf'].cpu().state_dict()
                    vf_cache = params['vf'].cpu().state_dict()
                    target_vf_cache = params['target_vf'].cpu().state_dict()
                    alpha_cache = params['alpha'].cpu()
                    policy_cache = params['policy'].cpu().state_dict()
                    joblib.dump({'policy': policy_cache, 'qf': qf_cache, 'vf': vf_cache, 'target_vf': target_vf_cache,
                                 'alpha': alpha_cache},
                                cache_file_name)
                    optim_file_name = osp.join(self._snapshot_dir, 'optim_itr_%d.pkl' % itr)
                    qf_optim = params['qf_optim'].state_dict()
                    vf_optim = params['vf_optim'].state_dict()
                    policy_optim = params['policy_optim'].state_dict()
                    alpha_optim = params['alpha_optim'].state_dict()
                    joblib.dump({'qf_optim': qf_optim, 'vf_optim': vf_optim,
                                 'policy_optim': policy_optim, 'alpha_optim': alpha_optim}, optim_file_name)
                    params['policy'].to(orig_run_device)
                    params['qf'].to(orig_run_device)
                    params['vf'].to(orig_run_device)
                    params['target_vf'].to(orig_run_device)
                    params['alpha'].to(orig_run_device)
                file_name = osp.join(self._snapshot_dir, 'params.pkl')
                joblib.dump(params, file_name)
                eval_file_name = osp.join(self._snapshot_dir, 'eval_params.pkl')
                mdl = params['eval_policy']
                orig_run_device = mdl.stochastic_policy.device
                run_device = 'cpu'
                mdl.stochastic_policy.device = run_device
                mdl.stochastic_policy = mdl.stochastic_policy.to(run_device)
                if mdl.stochastic_policy.features is not None:
                    mdl.stochastic_policy.features.device = run_device
                    mdl.stochastic_policy.features = mdl.stochastic_policy.features.to(run_device)
                joblib.dump(mdl, eval_file_name)
                mdl.stochastic_policy.device = orig_run_device
                mdl.stochastic_policy = mdl.stochastic_policy.to(orig_run_device)
                if mdl.stochastic_policy.features is not None:
                    mdl.stochastic_policy.features.device = orig_run_device
                    mdl.stochastic_policy.features = mdl.stochastic_policy.features.to(orig_run_device)
                cache_file_name = osp.join(self._snapshot_dir, 'cache.pkl')
                policy_cache = params['policy'].cpu().state_dict()
                qf_cache = params['qf'].cpu().state_dict()
                vf_cache = params['vf'].cpu().state_dict()
                target_vf_cache = params['target_vf'].cpu().state_dict()
                alpha_cache = params['alpha'].cpu()
                joblib.dump({'policy': policy_cache, 'qf': qf_cache, 'vf': vf_cache, 'target_vf': target_vf_cache,
                             'alpha': alpha_cache},
                            cache_file_name)
                optim_file_name = osp.join(self._snapshot_dir, 'optim.pkl')
                qf_optim = params['qf_optim'].state_dict()
                vf_optim = params['vf_optim'].state_dict()
                policy_optim = params['policy_optim'].state_dict()
                alpha_optim = params['alpha_optim'].state_dict()
                joblib.dump({'qf_optim': qf_optim, 'vf_optim': vf_optim,
                             'policy_optim': policy_optim, 'alpha_optim': alpha_optim}, optim_file_name)
                params['policy'].to(orig_run_device)
                params['qf'].to(orig_run_device)
                params['vf'].to(orig_run_device)
                params['target_vf'].to(orig_run_device)
                params['alpha'].to(orig_run_device)
                pass
            else:
                raise NotImplementedError


logger = Logger()
