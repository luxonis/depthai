if ! command -v snakeviz &> /dev/null
then
    echo -e "snakeviz module could not be found, run:\033[1;31m python3 -m pip install snakeviz \033[0m"
    exit 1
fi
python3 -m cProfile -o depthai.prof -s tottime depthai_demo.py "$@"
snakeviz depthai.prof
