import os
import stat
import shutil
from pathlib import Path
from datetime import datetime
import sys, os
import uuid
import errno
import tempfile

os.environ['TORCH_COMPILE_DISABLE'] = '1'

def configure_aimnet_cache(data_dir=None):
    """Keep model downloads writable, retaining access to preloaded weights."""
    data_dir = Path.cwd() if data_dir is None else Path(data_dir)
    fallback = data_dir.resolve().parent / '.cache' / 'aimnet'
    inherited = os.environ.get('AIMNET_CACHE_DIR')
    cache = Path(inherited).expanduser().resolve() if inherited else fallback
    try:
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=cache):
            pass
    except OSError as exc:
        if exc.errno not in (errno.EACCES, errno.EPERM, errno.EROFS) or cache == fallback:
            raise
        readonly_cache = cache
        cache = fallback
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=cache):
            pass
        # The image's immutable weights stay usable without copying or downloading.
        for model in readonly_cache.glob('*'):
            if model.is_file() and model.suffix in ('.pt', '.jpt'):
                try:
                    (cache / model.name).symlink_to(model)
                except FileExistsError:
                    pass  # Preserve weights already downloaded into this cache.
    os.environ['AIMNET_CACHE_DIR'] = str(cache)
    return cache


def start_exercise():
    """Configure a writable model cache and create a fresh output directory."""
    data = Path.cwd()
    cache = configure_aimnet_cache(data)
    output = data / 'results' / (datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    print(f'Python: {sys.executable}\nInputs: {data}\nAIMNet cache: {cache}\nNew results: {output}')
    return data, output

def configure_cli(target_dir='.', templates_dir=None):
    container = os.environ['SINGULARITY_CONTAINER'] # error out early
    name = os.path.splitext(os.path.basename(container))[0]
    script_name = f'run_{name}_sif.sh'
    script = os.path.join(target_dir,script_name)

    from McUtils.ExternalPrograms import SLURMClient
    SLURMClient.create_server_package(target_dir, overwrite=True)

    if templates_dir is None:
        templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    with open(os.path.join(templates_dir, 'container_runner.sh'), 'r') as f:
        src = f.read()
        src = src.replace('`CONTAINER_PATH`', container)
        with open(script, 'w+') as dest:
            dest.write(src)

    sbatch_name = f'{name}_sif_python.sh'
    sbatch_path = os.path.join(target_dir, sbatch_name)
    with open(os.path.join(templates_dir, 'container_sbatch.sh'), 'r') as f:
        src = f.read()
        src = src.replace('`CONTAINER_SCRIPT_NAME`', script_name)
        with open(sbatch_path, 'w+') as dest:
            dest.write(src)

    st = os.stat(script)
    os.chmod(script, st.st_mode | stat.S_IEXEC)

    return script

def configure_demo(target_dir='.', demo_dir='/home/demo'):
    targ = os.path.join(target_dir, 'demo')
    if not os.path.exists(targ):
        shutil.copytree(demo_dir, targ)

if __name__ == "__main__":
    import sys
    # configure_cli(*sys.argv[1:])
    configure_demo(*sys.argv[1:])