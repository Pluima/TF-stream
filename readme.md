# How to use

1. install the requirements
2. list input/output device IDs:
   ```bash
   python detect_devices.py
   ```
3. run the default streaming separator:
   ```bash
   python run_streaming.py --input-device X --output-device Y --device cuda:N (add this if not macOS)
   ```
   Replace `X` and `Y` with the input/output IDs printed by `detect_devices.py`.
