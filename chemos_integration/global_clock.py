import os
import json
import time
import threading
from typing import Optional


class GlobalClock:
    def __init__(self, time_scale: float = 0.1,
                 state_file: Optional[str] = None,
                 epoch: int = 0):
        self.time_scale = float(time_scale)
        self._base_min = 0.0
        self._anchor_real = time.time()
        self.lock = threading.Lock()
        self.state_file = state_file
        self.epoch = int(epoch)
        self._paused = False
        self._paused_min = 0.0
        self._publish()
        print(f"[GlobalClock] init scale={time_scale} epoch={epoch} "
              f"(1 virt min = {60 * time_scale:.2f}s real)")

    def _now_min(self) -> float:
        if self._paused:
            return self._paused_min
        return self._base_min + (time.time() - self._anchor_real) / self.time_scale / 60.0

    def get_current_time(self) -> float:
        with self.lock:
            return self._now_min() * 60.0

    def get_current_time_minutes(self) -> float:
        with self.lock:
            return self._now_min()

    def advance_to(self, target_seconds: float):
        target_min = target_seconds / 60.0
        with self.lock:
            cur_min = self._now_min()
        if target_min <= cur_min:
            return
        need_real = (target_min - cur_min) * 60.0 * self.time_scale
        if need_real > 0:
            time.sleep(need_real)

    def add_delay(self, delay_seconds: float):
        delay_min = delay_seconds / 60.0
        with self.lock:
            self._base_min += delay_min
            now = self._now_min()
        self._publish()
        print(f"[GlobalClock] +solver {delay_seconds:.3f}s "
              f"({delay_min:.4f}min)  now={now:.3f}min")

    def pause(self):
        with self.lock:
            if self._paused:
                return
            self._paused_min = self._base_min + (time.time() - self._anchor_real) / self.time_scale / 60.0
            self._paused = True
        self._publish()

    def resume_with_delay(self, real_seconds: float):
        delay_min = float(real_seconds) / 60.0
        with self.lock:
            base = self._paused_min if self._paused else (
                self._base_min + (time.time() - self._anchor_real) / self.time_scale / 60.0
            )
            self._base_min = base + delay_min
            self._anchor_real = time.time()
            self._paused = False
            self._paused_min = 0.0
            now = self._base_min
        self._publish()
        print(f"[GlobalClock] +cpu {real_seconds:.3f}s "
              f"({delay_min:.4f}min, 1:1)  now={now:.3f}min")

    def reset(self):
        with self.lock:
            self._base_min = 0.0
            self._anchor_real = time.time()
        self._publish()
        print("[GlobalClock] reset")

    def _publish(self):
        if not self.state_file:
            return
        try:
            with self.lock:
                payload = {
                    'base_min': self._base_min,
                    'anchor_real': self._anchor_real,
                    'time_scale': self.time_scale,
                    'epoch': self.epoch,
                    'ts': time.time(),
                    'paused': self._paused,
                    'paused_min': self._paused_min,
                }
            tmp = self.state_file + '.tmp'
            d = os.path.dirname(self.state_file)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(payload, f)
            last_err = None
            for attempt in range(50):
                try:
                    os.replace(tmp, self.state_file)
                    last_err = None
                    break
                except PermissionError as e:
                    last_err = e
                    time.sleep(0.01)
            if last_err is not None:
                try:
                    with open(self.state_file, 'w', encoding='utf-8') as f:
                        json.dump(payload, f)
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                except Exception as e2:
                    print(f"[GlobalClock] publish fallback also failed: {e2}")
        except Exception as e:
            print(f"[GlobalClock] publish failed: {e}")

    def get_stats(self):
        with self.lock:
            return {
                'virtual_time_seconds': self._now_min() * 60.0,
                'virtual_time_minutes': self._now_min(),
                'time_scale': self.time_scale,
                'real_elapsed_seconds': time.time() - self._anchor_real,
                'epoch': self.epoch,
            }

    def __repr__(self):
        s = self.get_stats()
        return (f"GlobalClock(t_abs={s['virtual_time_minutes']:.2f}min, "
                f"scale={self.time_scale}, ep={self.epoch})")
