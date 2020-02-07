logSystemInformation="log_system_information.txt"
uname -a | tee -a $logSystemInformation
python3 -c "import depthai; import numpy; print('depthai.__version__ == %s' % depthai.__version__); print('depthai.__dev_version__ == %s' % depthai.__dev_version__); print('numpy.__version__ == %s' % numpy.__version__);" | tee -a $logSystemInformation
python3 --version | tee -a $logSystemInformation
