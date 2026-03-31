# How to use

1. modify the SYSTEM_LIBSTDCXX path if necessary
2. install the requirements
3. modify cuda if not MACOS (line 538 in run_streaming.py)
4. python stereo_capture.py --list-devices
5. python run_streaming.py --input-device X --output-device Y
   (Replace X and Y for the devices listed above)
