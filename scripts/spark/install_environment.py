"""Install the official ARM64 CUDA wheel with verified hash and IPv4 transport.

IPv4 preference is limited to this installer process. TLS remains verified.
No system Python packages or network settings are changed.
"""
import hashlib
import os
from html.parser import HTMLParser
from pathlib import Path
import platform
import socket
import subprocess
import sys
from urllib.parse import unquote, urlsplit, urlunsplit

ROOT=Path(__file__).resolve().parents[2]
VERSION='2.9.1'


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls=[]

    def handle_starttag(self,tag,attrs):
        if tag=='a':
            href=dict(attrs).get('href')
            if href:self.urls.append(href)


def main():
    if platform.system()!='Linux' or platform.machine() not in ('aarch64','arm64'):
        raise RuntimeError('Installer requires ARM64 Linux')
    from pip._vendor.urllib3.util import connection
    connection.allowed_gai_family=lambda:socket.AF_INET
    os.environ['PIP_CACHE_DIR']=str(ROOT/'.cache/pip')
    from pip._vendor import requests
    from pip._internal.cli.main import main as pip_main
    index=requests.get('https://download.pytorch.org/whl/cu130/torch/',timeout=30)
    index.raise_for_status()
    parser=Links();parser.feed(index.text)
    abi=f'cp{sys.version_info.major}{sys.version_info.minor}'
    filename=f'torch-{VERSION}+cu130-{abi}-{abi}-manylinux_2_28_aarch64.whl'
    matches=[url for url in parser.urls if unquote(urlsplit(url).path.rsplit('/',1)[-1])==filename]
    if len(matches)!=1:raise RuntimeError(f'No unique official wheel for {filename}')
    parts=urlsplit(matches[0])
    if parts.scheme!='https' or parts.hostname not in ('download.pytorch.org','download-r2.pytorch.org'):
        raise RuntimeError('Unexpected wheel origin')
    if not parts.fragment.startswith('sha256='):raise RuntimeError('Official wheel hash missing')
    expected=parts.fragment.removeprefix('sha256=')
    url=urlunsplit(('https','download.pytorch.org',parts.path,parts.query,''))
    wheel=ROOT/'.cache/wheels'/filename;wheel.parent.mkdir(parents=True,exist_ok=True)
    existing_hash=None
    if wheel.exists():
        with wheel.open('rb') as stream:
            existing_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    if existing_hash!=expected:
        temporary=wheel.with_suffix('.part')
        subprocess.run(['curl','-4','-fL','--connect-timeout','10','--retry','2',url,'-o',str(temporary)],check=True)
        with temporary.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=expected:
                raise RuntimeError('CUDA wheel SHA256 mismatch')
        temporary.replace(wheel)
    raise SystemExit(pip_main(['install',str(wheel),'-r',str(ROOT/'requirements-rt-sys.txt'),
        '--index-url','https://pypi.org/simple']))


if __name__=='__main__':main()
