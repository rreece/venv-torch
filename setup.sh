#
# Script for installing or activating a virtualenv
#


#-----------------------------------------------------------------------------
# pre-setup helpers, don't touch
#-----------------------------------------------------------------------------

path_of_this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
path_above_this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"

add_to_path()
{
    export PATH=$1${PATH:+:${PATH}}
    echo "  Added $1 to your PATH."
}

add_to_ld_library_path()
{
    export LD_LIBRARY_PATH=$1${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    echo "  Added $1 to your LD_LIBRARY_PATH."
}

add_to_python_path()
{
    export PYTHONPATH=$1${PYTHONPATH:+:${PYTHONPATH}}
    echo "  Added $1 to your PYTHONPATH."
}


#-----------------------------------------------------------------------------
# setup virtualenv
#-----------------------------------------------------------------------------

venv_name=".venv"

if [ -f ${venv_name}/bin/activate ]; then
    source ${venv_name}/bin/activate
else
    echo "  Setting up virtualenv ${venv_name}"
    python -m venv ${venv_name}
    source ${venv_name}/bin/activate
    pip install -r requirements.txt
fi


#-----------------------------------------------------------------------------
# setup paths
#-----------------------------------------------------------------------------

add_to_path /usr/local/cuda-11.8/bin
add_to_ld_library_path /usr/local/cuda-11.8/lib64
add_to_python_path ${path_of_this_dir}

echo "  Done."
