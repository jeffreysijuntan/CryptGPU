import concurrent.futures
import configparser
import os
import sys
import time
import uuid
import warnings
from argparse import REMAINDER, ArgumentParser
from pathlib import Path

import boto3
import paramiko



def connect_to_instance(hostname, username, password, http_proxy=None):
    print("Connecting to {}@{} ...".format(username, hostname))

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    retries = 20

    while retries > 0:
        try:
            client.connect(
                hostname=hostname,
                username=username,
                password=password,
                timeout=10,
                sock=proxy if http_proxy else None,
            )
            print("Connected to {}@{}".format(username, hostname))
            break
        except Exception as e:
            print(f"Exception: {e} Retrying...")
            retries -= 1
            time.sleep(10)
    return client

def run_command(hostname, client, cmd, environment=None, inputs=None):
    stdin, stdout, stderr = client.exec_command(
        cmd, get_pty=True, environment=environment
    )
    if inputs:
        for inp in inputs:
            stdin.write(inp)

    def read_lines(fin, fout, line_head):
        line = ""
        while not fin.channel.exit_status_ready():
            line += fin.read(1).decode("utf8")
            if line.endswith("\n"):
                print(f"{line_head}{line[:-1]}", file=fout)
                line = ""
        if line:
            # print what remains in line buffer, in case fout does not
            # end with '\n'
            print(f"{line_head}{line[:-1]}", file=fout)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as printer:
        printer.submit(read_lines, stdout, sys.stdout, f"[{hostname} STDOUT] ")
        printer.submit(read_lines, stderr, sys.stderr, f"[{hostname} STDERR] ")


def upload_file(hostname, client, localpath, remotepath):
    ftp_client = client.open_sftp()
    print(f"Uploading `{localpath}` to {hostname}...")
    ftp_client.put(localpath, remotepath)
    ftp_client.close()
    print(f"`{localpath}` uploaded to {hostname}.")


def parse_args():
    """
    Helper function parsing the command line options
    """
    parser = ArgumentParser(
        description="PyTorch distributed training launch "
        "helper utilty that will spawn up "
        "parties for MPC scripts on AWS"
    )

    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        help="The port used by master instance " "for distributed training",
    )

    parser.add_argument(
        "--aux_files",
        type=str,
        default=None,
        help="The comma-separated paths of additional files "
        " that need to be transferred to AWS instances. "
        "If more than one file needs to be transferred, "
        "the basename of any two files can not be the "
        "same.",
    )

    parser.add_argument(
        "--prepare_cmd",
        type=str,
        default="",
        help="The command to run before running distribute "
        "training for prepare purpose, e.g., setup "
        "environment, extract data files, etc.",
    )

    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default="",
    )

    # positional
    parser.add_argument(
        "training_script",
        type=str,
        help="The full path to the single machine training "
        "program/script to be launched in parallel, "
        "followed by all the arguments for the "
        "training script",
    )

    # rest from the training program
    parser.add_argument("training_script_args", nargs=REMAINDER)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    username = "root"
    password = "password"
    hostnames = [
        "example1.server.com",
        "example2.server.com",
        "example3.server.com",
    ]
    master_ip_address = "1.1.1.1"

    client_dict = {}
    for hostname in hostnames:
        client = connect_to_instance(hostname, username, password)
        client_dict[hostname] = client


    world_size = 3
    print(f"Running world size {world_size}")

    assert os.path.exists(
        args.training_script
    ), f"File `{args.training_script}` does not exist"
    file_paths = args.aux_files.split(",") if args.aux_files else []
    for local_path in file_paths:
        assert os.path.exists(local_path), f"File `{local_path}` does not exist"

    remote_dir = f"crypten-launcher-tmp-{uuid.uuid1()}"
    script_basename = os.path.basename(args.training_script)
    remote_script = os.path.join(remote_dir, script_basename)

    # Upload files to all instances concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as uploaders:
        for hostname, client in client_dict.items():
            run_command(hostname, client, f"mkdir -p {remote_dir}")
            uploaders.submit(
                upload_file, hostname, client, args.training_script, remote_script
            )
            for local_path in file_paths:
                uploaders.submit(
                    upload_file,
                    hostname,
                    client,
                    local_path,
                    os.path.join(remote_dir, os.path.basename(local_path)),
                )
    for hostname, client in client_dict.items():
        run_command(hostname, client, f"chmod +x {remote_script}")
        run_command(hostname, client, f"ls -al {remote_dir}")

    environment = {
        "WORLD_SIZE": str(world_size),
        "RENDEZVOUS": "env://",
        #"RENDEZVOUS": "file:///tmp/crypten-rendezvous-{}".format(uuid.uuid1()),
        "MASTER_ADDR" : master_ip_address,
        "MASTER_PORT": str(args.master_port),
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=world_size+1) as executor:
        rank = 0
        for hostname, client in client_dict.items():
            environment["RANK"] = str(rank)
            # TODO: Although paramiko.SSHClient.exec_command() can accept
            # an argument `environment`, it seems not to take effect in
            # practice. It might because "Servers may silently reject
            # some environment variables" according to paramiko document.
            # As a workaround, here all environment variables are explicitly
            # exported.
            environment_cmd = "; ".join(
                [f"export {key}={value}" for (key, value) in environment.items()]
            )
            prepare_cmd = f"{args.prepare_cmd}; " if args.prepare_cmd else ""
            cmd = "{}; {} {} {} {}".format(
                environment_cmd,
                f"cd {remote_dir} ;",
                prepare_cmd,
                f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} ./{script_basename}",
                " ".join(args.training_script_args),
            )
            print(f"Run command: {cmd}")
            executor.submit(run_command, hostname, client, cmd, environment)
            rank += 1

    # Cleanup temp dir.
    for hostname, client in client_dict.items():
        run_command(hostname, client, f"rm -rf {remote_dir}")
        client.close()
