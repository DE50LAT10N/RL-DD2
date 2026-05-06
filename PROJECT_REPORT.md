# Отчет по проекту RL DD

Дата: 6 мая 2026

## Назначение проекта

Проект реализует исследовательского RL-агента для пошаговых боев Darkest Dungeon II. Основная идея: обучать политику в локальном симуляторе боя, а затем использовать сохраненную PPO-модель в live-режиме через BepInEx-мод, который читает состояние игры и принимает команды по локальному TCP/NDJSON-протоколу.

## Текущий состав проекта

- `env/` - симулятор боя, модель данных, Gymnasium-среда и расчет награды.
- `agents/` - обертка над PPO-моделью и базовые агенты.
- `scripts/` - обучение, staged-запуски, evaluation, live runner, валидация данных и smoke-тесты.
- `configs/` - настройки награды, party lineup, протокол мода и data overrides.
- `tests/fixtures/game_data/` - тестовые данные героев, монстров, предметов, токенов и encounters.
- `mod/` - BepInEx Mono plugin для DD2 и Python-клиент live-интеграции.
- `notebooks/training_dashboard.ipynb` - ноутбук запуска и анализа обучения.
- `runs/best/best_model.zip` - текущий сохраненный артефакт модели.

## Использованные методы

### 1. Формализация боя как MDP/Gymnasium-среды

Реализация: `env/dd_env.py`, `env/data_model.py`, `env/engine.py`.

Метод:

- бой представлен как среда `DarkestDungeonEnv`;
- состояние кодируется в числовой observation vector;
- действия представлены дискретным action space;
- эпизод завершается победой, поражением или truncation по `max_episode_steps`.

Причина выбора:

- Gymnasium является стандартным интерфейсом для RL-задач;
- совместимость с Stable-Baselines3 и sb3-contrib позволяет использовать готовые реализации PPO;
- дискретное пространство хорошо соответствует пошаговому бою: skill, item, move, pass.

Результат:

- observation space включает HP, stress, rank, alive/afflicted/remnant-флаги, токены, round/turn, предметы, encounter hint и relationships;
- action space включает 43 действия: навыки по целям, предметы, перемещения и pass;
- среда совместима с MaskablePPO через `action_masks()`.

### 2. Симулятор боевой логики

Реализация: `env/engine.py`.

Метод:

- локальный пошаговый backend `SimBattleBackend`;
- моделирование инициативы, скорости, рангов, cooldown, DOT, death's door, remnant/tombstone, токенов, guard/taunt, block/dodge/riposte, stun, stress и affliction;
- враги управляются эвристикой выбора цели и reposition.

Причина выбора:

- обучение напрямую в игре слишком медленное и нестабильное;
- симулятор дает быстрые миллионы шагов и контролируемую повторяемость по seed;
- достаточно подробная механика нужна, чтобы политика не училась на слишком абстрактной задаче.

Результат:

- симулятор поддерживает основные боевые эффекты, нужные для PPO-тренировки;
- добавлены явные move/pass-действия и remnant-логика;
- детерминированные smoke-тесты покрывают block, dodge, riposte, death's door/remnant, move/pass и stress meltdown.

### 3. Маскирование недоступных действий

Реализация: `DarkestDungeonEnv.action_masks()`, `_encode_action()`, `_decode_action()`, `scripts/live_ppo.py`.

Метод:

- на каждом ходе строится boolean-mask только для легальных действий;
- нелегальные индексы скрываются от политики MaskablePPO;
- в live-режиме mask строится из `legal_actions`, присланных модом.

Причина выбора:

- в DD2 большая часть действий на конкретном ходу недоступна из-за ранга, цели, cooldown, стороны цели или статуса персонажа;
- обычный PPO тратил бы много выборов на невозможные действия;
- маскирование повышает стабильность обучения и уменьшает необходимость штрафовать invalid actions.

Результат:

- среда и live runner используют одну концепцию action mask;
- если mask пустая, fallback переводит действие в pass;
- live runner дополнительно банит действия, которые в игре дали ack без изменения состояния.

### 4. Reward shaping

Реализация: `env/rewards.py`, `configs/reward.yaml`.

Метод:

- терминальные награды за победу/поражение;
- плотная награда за урон врагам, штраф за урон героям, сохраненный stress, убийства врагов, живых героев, affliction, смерть героя;
- potential-based shaping через функцию `phi`;
- дополнительные бонусы/штрафы за лечение, защитные токены, pass, move, stall, no-effect skill, diversity и повторение одного действия.

Причина выбора:

- победа/поражение слишком редкий сигнал для долгих боев;
- плотная награда ускоряет обучение тактических решений;
- отдельные поправки нужны, чтобы агент не застревал в pass/stall-поведении и учился лечить/защищать в критических состояниях.

Результат:

- веса вынесены в `configs/reward.yaml`;
- награда стала настраиваемой без изменения кода;
- в текущей конфигурации усилены победа, убийства, лечение low-HP целей, защита и штрафы за бесполезные/повторные действия.

### 5. MaskablePPO

Реализация: `agents/ppo_agent.py`, `scripts/train.py`.

Метод:

- используется `sb3_contrib.MaskablePPO`;
- политика `MlpPolicy`;
- поддерживаются CUDA/CPU, vectorized envs, learning-rate schedule, entropy coefficient, checkpointing и eval callback.

Причина выбора:

- PPO устойчив для дискретных управляемых сред;
- MaskablePPO напрямую использует action masks;
- Stable-Baselines3 дает готовую инфраструктуру сохранения, evaluation callback, TensorBoard и возобновления обучения.

Результат обучения из `notebooks/training_dashboard.ipynb`:

- основной запуск: `10_000_000` timesteps;
- параметры запуска: `16` envs, CUDA, learning rate `0.00025`, `lr_end_ratio=0.08`, `ent_coef=0.01`, `net_arch=512,512`, `n_steps=2048`, `batch_size=1024`, `n_epochs=6`, `gamma=0.995`, `max_episode_steps=120`;
- eval callback выполнялся каждые `100_000` timesteps;
- лучший зафиксированный eval callback: `episode_reward=86.84 +/- 19.17` на `2_100_000` timesteps;
- финальная зафиксированная eval-точка на `10_000_000` timesteps: `episode_reward=76.36 +/- 38.36`;
- текущий доступный артефакт модели: `runs/best/best_model.zip`, размер `6,014,488` байт.

### 6. Curriculum learning

Реализация: `CurriculumEnv` в `scripts/train.py`.

Метод:

- encounters вводятся фазами: сначала простые дорожные бои, затем расширенный пул, затем holdout/elite encounters;
- сложные encounters oversample'ятся, если win rate по ним ниже;
- доля easy encounters динамически меняется по недавним результатам.

Причина выбора:

- сразу обучаться на elite/holdout-сценариях сложнее и менее стабильно;
- curriculum помогает сначала освоить базовый бой, а затем переносить навык на более трудные варианты;
- oversampling слабых мест направляет обучение на encounters, где политика чаще проигрывает.

Результат:

- обучение дошло до полного пула, включая `holdout_elite_pair` и `holdout_elite_swarm`;
- сохранены callback-метрики в ноутбуке;
- staged-тренировка также поддерживается отдельным скриптом `scripts/train_staged.py`.

### 7. Holdout evaluation

Реализация: `scripts/evaluate.py`.

Метод:

- модель проверяется на фиксированном наборе holdout encounters;
- считаются win rate, mean reward, mean steps, survival rate;
- поддерживается несколько seed и JSON-вывод.

Причина выбора:

- holdout-сценарии отделяют оценку от текущей curriculum-смеси;
- несколько seed уменьшают зависимость результата от одного случайного запуска;
- JSON-вывод удобен для построения графиков и сравнения моделей.

Результат:

- в ноутбуке зафиксирован успешный запуск evaluation-команды с `returncode=0`;
- использовались `200` эпизодов на seed и seeds `17,27,37`;
- детальный JSON `runs/dd2_ppo_10m_v1/holdout_eval.json` относился к тренировочным артефактам run-директории и сейчас отсутствует после чистки ignored outputs.

### 8. Данные и overrides

Реализация: `env/game_data.py`, `configs/data_overrides/*.json`, `tests/fixtures/game_data/`.

Метод:

- данные загружаются из `%APPDATA%/DDRL/data`;
- при отсутствии внешних данных используются defaults;
- локальные overrides расширяют героев, encounters и elite-монстров;
- Pydantic-модели валидируют структуру.

Причина выбора:

- проект не должен распространять proprietary game assets;
- defaults и fixtures позволяют запускать тесты без извлеченных игровых данных;
- overrides дают контролируемую тренировочную постановку.

Результат:

- задан основной party lineup: Man-at-Arms, Hellion, Plague Doctor, Highwayman;
- добавлены holdout encounters и elite-монстры;
- CI проверяет загрузку fixture-данных командой `scripts/validate_game_data.py`.

### 9. Live-интеграция с DD2

Реализация: `mod/`, `mod/agent/`, `env/live/mod_state.py`, `scripts/live_ppo.py`, `scripts/run_live_game.ps1`.

Метод:

- BepInEx 5 Mono plugin работает внутри DD2;
- plugin публикует state snapshots по localhost `127.0.0.1:8765`;
- протокол основан на JSON lines: `hello`, `state`, `action`, `ack`, `battle_end`, `ping`, `pong`, `error`;
- Python runner преобразует live state в формат `BattleState`, строит mask и отправляет payload действия.

Причина выбора:

- localhost NDJSON проще отлаживать и логировать, чем бинарный IPC;
- hook-based action path надежнее screen-click automation;
- разделение C# plugin и Python policy сохраняет RL-часть вне игрового процесса.

Результат:

- описан протокол в `configs/mod_protocol.json`;
- C# plugin содержит state reader, TCP server, dispatcher и Harmony hooks;
- live runner имеет защитные режимы `pass_only`, `hook_only`, `ppo`, ожидание hero turn, обработку stunned/remnant-ситуаций и safe-stop при stuck/no-ack.

### 10. Проверка качества

Реализация: `.github/workflows`, `scripts/validate_game_data.py`, `scripts/test_engine_deterministic.py`.

Метод:

- CI устанавливает зависимости, валидирует fixture-данные и запускает детерминированный smoke-тест движка;
- локальные тесты проверяют ключевые боевые эффекты.

Причина выбора:

- проект содержит симулятор с большим количеством правил, где легко получить скрытую регрессию;
- smoke-тесты быстрые и подходят для каждого push/pull request;
- fixture-валидация защищает формат данных от несовместимых изменений.

Результат:

- CI workflow настроен;
- локальная попытка запуска после чистки не выполнилась, потому что команда `python` сейчас указывает на недоступный WindowsApps launcher/частично удаленное окружение;
- для повторной локальной проверки нужно восстановить Python-окружение по `requirements.txt`.

## Общие результаты проекта

- Создана Gymnasium-среда для DD2-подобного боя.
- Реализован локальный симулятор с токенами, инициативой, ранговой системой, remnant/death's door, предметами и отношениями.
- Реализована MaskablePPO-тренировка с curriculum learning и holdout evaluation.
- Получена модель после крупного запуска на `10_000_000` timesteps.
- Зафиксирован лучший eval callback `86.84 +/- 19.17`, финальный eval callback `76.36 +/- 38.36`.
- Создан live bridge через BepInEx plugin и Python runner.
- Настроены CI-проверки для данных и детерминированной логики движка.

## Ограничения и риски

- Симулятор является приближением реальной DD2-механики, поэтому качество в live-режиме зависит от расхождения simulator/live state.
- Детальный holdout JSON из 10M run отсутствует в текущем дереве после удаления ignored run outputs; численные итоги доступны в notebook output.
- Текущее локальное Python-окружение нужно восстановить перед повторным запуском тестов, evaluation или notebook kernel.
- Live-интеграция зависит от конкретных DD2/BepInEx/Harmony class paths и может требовать обновления после патчей игры.

## Рекомендуемые следующие шаги

1. Восстановить окружение: создать новую `.venv` и установить `requirements.txt`.
2. Запустить `python scripts/validate_game_data.py --data-root ./tests/fixtures/game_data`.
3. Запустить `python scripts/test_engine_deterministic.py`.
4. Повторить `scripts/evaluate.py` для текущего `runs/best/best_model.zip` и сохранить свежий `holdout_eval.json` вне удаляемой run-директории или добавить явное исключение в `.gitignore`.
5. Для live-теста использовать guarded runner через `scripts/run_live_game.ps1`.
