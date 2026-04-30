"""Helpers for adapting Unitree SDK2 runtime config to the local workspace."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cyclonedds_supports_ignore_type_information() -> bool:
    try:
        version = metadata.version("cyclonedds")
    except Exception:
        return False

    major_text = version.split(".", maxsplit=1)[0]
    try:
        return int(major_text) >= 11
    except ValueError:
        return False


def cyclonedds_config_xml(
    interface: str | None = None,
    output_file: Path | None = None,
    verbosity: str = "config",
    ignore_type_information_vendor_ids: str | None = None,
) -> str:
    interface_xml = (
        f'<NetworkInterface name="{interface}" priority="default" multicast="default"/>'
        if interface
        else '<NetworkInterface autodetermine="true" priority="default" multicast="default"/>'
    )
    tracing_xml = ""
    if output_file is not None:
        tracing_xml = f"""
    <Tracing>
      <Verbosity>{verbosity}</Verbosity>
      <OutputFile>{output_file}</OutputFile>
    </Tracing>"""
    compatibility_xml = ""
    vendor_ids = ignore_type_information_vendor_ids
    if vendor_ids is None and cyclonedds_supports_ignore_type_information():
        vendor_ids = "1.16"
    if vendor_ids:
        compatibility_xml = """
    <Compatibility>
      <IgnoreTypeInformation>{vendor_ids}</IgnoreTypeInformation>
    </Compatibility>""".format(vendor_ids=vendor_ids)

    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
  <Domain Id="any">
    <General>
      <Interfaces>
        {interface_xml}
      </Interfaces>
    </General>{compatibility_xml}{tracing_xml}
  </Domain>
</CycloneDDS>"""


def configure_local_cyclonedds_log() -> Path:
    """Redirect Unitree SDK2's CycloneDDS trace log into the repo.

    The upstream SDK hardcodes `/tmp/cdds.LOG` in its interface-specific DDS
    config. Some environments do not allow writing to `/tmp`, which makes
    `ChannelFactoryInitialize(..., interface)` fail before discovery even
    starts. We patch the imported config strings in-process before the channel
    factory is initialized.
    """

    log_dir = project_root() / "runs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "cdds.log"
    from unitree_sdk2py.core import channel as channel_module
    from unitree_sdk2py.core import channel_config as channel_config_module

    interface_config = cyclonedds_config_xml(
        interface="$__IF_NAME__$",
        output_file=log_path,
    )
    auto_config = cyclonedds_config_xml(output_file=log_path)

    channel_config_module.ChannelConfigHasInterface = interface_config
    channel_config_module.ChannelConfigAutoDetermine = auto_config
    channel_module.ChannelConfigHasInterface = interface_config
    channel_module.ChannelConfigAutoDetermine = auto_config
    return log_path
