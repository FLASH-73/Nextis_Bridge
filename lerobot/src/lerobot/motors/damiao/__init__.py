# Copyright 2024 Nextis. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Damiao motor bus implementation for J-series motors (J8009P, J4340P, J4310)."""

from .damiao import DamiaoMotorsBus
from .tables import DAMIAO_MOTOR_SPECS

__all__ = ["DamiaoMotorsBus", "DAMIAO_MOTOR_SPECS"]
