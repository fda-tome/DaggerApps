#!/usr/bin/env bash
# ALCF HTTP/HTTPS proxy for Polaris (and other ALCF systems that use the same endpoint).
# Use when jobs or login shells need outbound internet: curl, wget, pip, Julia Pkg, git, etc.
#
# Official docs: https://docs.alcf.anl.gov/polaris/getting-started/
# Facility updates (URL changes): https://www.alcf.anl.gov/support-center/facility-updates
#
# Usage (from a fresh shell on Polaris):
#   source /path/to/DaggerApps/scripts/polaris_proxy_env.sh
#
# Optional: append the two export blocks below to ~/.bashrc on Polaris for persistence.

_POLARIS_PROXY_URL="${POLARIS_PROXY_URL:-http://proxy.alcf.anl.gov:3128}"

export HTTP_PROXY="${_POLARIS_PROXY_URL}"
export HTTPS_PROXY="${_POLARIS_PROXY_URL}"
export http_proxy="${_POLARIS_PROXY_URL}"
export https_proxy="${_POLARIS_PROXY_URL}"
export ftp_proxy="${_POLARIS_PROXY_URL}"
export ALL_PROXY="${_POLARIS_PROXY_URL}"
export all_proxy="${_POLARIS_PROXY_URL}"

# Do not send internal ALCF / node traffic through the outbound proxy.
export NO_PROXY="admin,localhost,127.0.0.1,polaris-adminvm-01,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"
export no_proxy="${NO_PROXY}"

echo "ALCF proxy env set: HTTP(S)_PROXY=${HTTP_PROXY}"
echo "To clear: unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy ALL_PROXY all_proxy NO_PROXY no_proxy"
