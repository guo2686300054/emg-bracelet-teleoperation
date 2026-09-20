import tempfile,unittest,warnings
from pathlib import Path
from app_config import ConfigError,MIN_RAW_AUDIT_FILE_BYTES,RAW_AUDIT_MAX_BYTES_LIMIT,RAW_AUDIT_TOTAL_BUDGET_LIMIT,RateConfig,RawAuditConfig,load_config
from emg_protocol import LOGICAL16_PROTOCOL,PADDED28_PROTOCOL
class ConfigTests(unittest.TestCase):
 def test_v1_rates_and_paths(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"c.ini"; p.write_text("[App]\nconfig_version=1.0\n[Data]\ndata_path=x\nadc_sampling_rate=1000\nadc_sampling_rate_source_kind=firmware\nadc_sampling_rate_evidence_ref=rev1\nadc_sampling_rate_confirmed=true\n",encoding="utf8")
   c=load_config(p); self.assertTrue(c.data.adc_sampling_rate.confirmed); self.assertEqual(c.data.data_path,(Path(d)/"x").resolve())
 def test_v0_migration_and_future_rejection(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"c.ini"; p.write_text("[Data]\nsampling_rate=50\nsave_path=old\n",encoding="utf8")
   with self.assertWarns(UserWarning): c=load_config(p)
   self.assertEqual(c.data.legacy_unclassified_rate.value_hz,50); self.assertFalse(c.data.legacy_unclassified_rate.confirmed)
   p.write_text("[App]\nconfig_version=2.0\n",encoding="utf8")
   with self.assertRaises(ConfigError): load_config(p)
 def test_unknown_keys_and_nonfinite_rejected(self):
  for body in ("[App]\nconfig_version=1.0\n[Data]\nbogus=1\n","[App]\nconfig_version=1.0\n[Data]\nadc_sampling_rate=nan\n"):
   with self.subTest(body=body),tempfile.TemporaryDirectory() as d:
    p=Path(d)/"c.ini";p.write_text(body,encoding="utf8")
    with self.assertRaises(ConfigError):load_config(p)
 def test_unknown_section_and_v0_typo_rejected(self):
  for body in ("[Mystery]\nx=1\n","[Data]\nsamplng_rate=50\n"):
   with tempfile.TemporaryDirectory() as d:
    p=Path(d)/"c.ini";p.write_text(body,encoding="utf8")
    with self.assertRaises(ConfigError):load_config(p)
 def test_empty_critical_values_and_negative_version(self):
  for body in ("[App]\nconfig_version=-1.0\n","[App]\nconfig_version=1.0\n[Serial]\nport=\n","[App]\nconfig_version=1.0\n[Data]\ndata_path=\n"):
   with tempfile.TemporaryDirectory() as d:
    p=Path(d)/"c.ini";p.write_text(body,encoding="utf8")
    with self.assertRaises(ConfigError):load_config(p)
 def test_missing_defaults_and_ranges(self):
  with tempfile.TemporaryDirectory() as d:
   c=load_config(Path(d)/"missing.ini");self.assertEqual(c.data.channels,8);self.assertIsNone(c.data.adc_sampling_rate.value_hz)
   p=Path(d)/"c.ini";p.write_text("[App]\nconfig_version=1.0\n[Data]\nchannels=257\n",encoding="utf8")
   with self.assertRaises(ConfigError):load_config(p)
 def test_log_ranges(self):
  for line in ("max_bytes=0","backup_count=-1","queue_capacity=0","level=NOPE"):
   with tempfile.TemporaryDirectory() as d:
    p=Path(d)/"c.ini";p.write_text("[App]\nconfig_version=1.0\n[Logging]\n"+line+"\n",encoding="utf8")
    with self.assertRaises(ConfigError):load_config(p)
 def test_default_keys_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"c.ini";p.write_text("[DEFAULT]\nsecret=x\n",encoding="utf8")
   with self.assertRaises(ConfigError):load_config(p)
 def test_rate_schema_rejects_bool_negative_string_and_unproven_confirmation(self):
  for args in ((True,"source","evidence",False),(-1,"source","evidence",False),("50","source","evidence",False),(50," unknown ","evidence",True),(50,"source"," unknown ",True),(50,"source","evidence","true")):
   with self.subTest(args=args),self.assertRaises(ValueError):RateConfig(*args)
  self.assertEqual(RateConfig(50,"firmware","rev-1",True).value_hz,50)
 def test_confirmed_ini_rate_rejects_whitespace_unknown_evidence(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"c.ini";p.write_text("[App]\nconfig_version=1.0\n[Data]\nadc_sampling_rate=1000\nadc_sampling_rate_source_kind= unknown \nadc_sampling_rate_evidence_ref=rev\nadc_sampling_rate_confirmed=true\n",encoding="utf8")
   with self.assertRaises(ConfigError):load_config(p)
 def test_future_minor_and_unrecognized_legacy_minor_are_rejected(self):
  for version in ("1.5","0.1","0.99"):
   with self.subTest(version=version),tempfile.TemporaryDirectory() as d:
    p=Path(d)/"c.ini";p.write_text(f"[App]\nconfig_version={version}\n",encoding="utf8")
    with self.assertRaises(ConfigError):load_config(p)
 def test_config_converts_to_domain_metadata_without_rate_duplication(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"c.ini";p.write_text("[App]\nconfig_version=1.0\n[BLE]\ndevice_id=dev-0123456789abcdef0123456789abcdef\n[Data]\nsample_format=uint8\nunit=adc_code\nadc_sampling_rate=1000\nadc_sampling_rate_source_kind=firmware\nadc_sampling_rate_evidence_ref=rev1\nadc_sampling_rate_confirmed=true\n",encoding="utf8")
   config=load_config(p);metadata=config.to_acquisition_metadata()
   self.assertIs(config.data.adc_sampling_rate,metadata.adc_rate);self.assertIsNone(metadata.sample_rate.value_hz);self.assertEqual(metadata.signal_chain.sample_format,"uint8");self.assertEqual(str(config.ble.device_key),"dev-0123456789abcdef0123456789abcdef")
 def test_explicit_sample_time_rate_is_the_only_sample_clock(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"c.ini";p.write_text("[App]\nconfig_version=1.0\n[Data]\nsample_time_rate=200\nsample_time_rate_source_kind=protocol\nsample_time_rate_evidence_ref=spec-2\nsample_time_rate_confirmed=true\nadc_sampling_rate=1000\nadc_sampling_rate_source_kind=firmware\nadc_sampling_rate_evidence_ref=rev1\nadc_sampling_rate_confirmed=true\ndevice_output_rate=50\ndevice_output_rate_source_kind=observed\ndevice_output_rate_evidence_ref=run1\ndevice_output_rate_confirmed=true\n",encoding="utf8")
   metadata=load_config(p).to_acquisition_metadata()
   self.assertEqual(metadata.sample_rate.value_hz,200);self.assertEqual(metadata.adc_rate.value_hz,1000);self.assertEqual(metadata.device_output_rate.value_hz,50)
 def test_legacy_rate_never_becomes_sample_clock(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"c.ini";p.write_text("[Data]\nsampling_rate=50\n",encoding="utf8")
   with self.assertWarns(UserWarning):metadata=load_config(p).to_acquisition_metadata()
   self.assertIsNone(metadata.sample_rate.value_hz)
 def test_control_characters_and_ini_continuations_are_rejected(self):
  bodies=(
   "[App]\nconfig_version=1.0\n[BLE]\ndevice_name=bracelet\n continued\n",
   "[App]\nconfig_version=1.0\n[Data]\nsample_time_rate_source_kind=firmware\n continued\n",
   "[App]\nconfig_version=1.0\n[Data]\nsample_time_rate_evidence_ref=spec\tref\n",
  )
  for body in bodies:
   with self.subTest(body=body),tempfile.TemporaryDirectory() as d:
    p=Path(d)/"c.ini";p.write_text(body,encoding="utf8")
    with self.assertRaises(ConfigError):load_config(p)
 def test_protocol_provenance_rejects_control_characters(self):
  from emg_protocol import AcquisitionMetadata,RateDescriptor,SignalChain
  for factory in (
   lambda:RateDescriptor(10,"firmware\ncontinued","spec",False),
   lambda:SignalChain(sample_format="uint8",unit="adc\tcode"),
   lambda:AcquisitionMetadata(device_tick_modulus_source="wire\rsource"),
  ):
   with self.assertRaises(ValueError):factory()
 def test_repository_config_is_current_and_domain_valid(self):
  config=load_config(Path(__file__).resolve().parents[1]/"config.ini");metadata=config.to_acquisition_metadata();self.assertEqual(config.config_version,"1.4");self.assertTrue(config.raw_audit.enabled);self.assertEqual(config.packet_protocol.mode,PADDED28_PROTOCOL);self.assertEqual(config.packet_protocol.wire_packet_size,28);self.assertEqual(metadata.signal_chain.sample_format,"uint8");self.assertIsNone(metadata.device_sequence_modulus)
 def test_sequence_modulus_requires_bidirectional_provenance_consistency(self):
  bodies=(
   "[App]\nconfig_version=1.1\n[Data]\ndevice_sequence_modulus=256\n",
   "[App]\nconfig_version=1.1\n[Data]\ndevice_sequence_modulus=unknown\ndevice_sequence_modulus_source=protocol\n",
   "[App]\nconfig_version=1.1\n[Data]\ndevice_sequence_modulus=1\ndevice_sequence_modulus_source=protocol\n",
  )
  for body in bodies:
   with self.subTest(body=body),tempfile.TemporaryDirectory() as d:
    p=Path(d)/"c.ini";p.write_text(body,encoding="utf8")
    with self.assertRaises(ConfigError):load_config(p)
 def test_adc_narrowing_requires_explicit_quantization(self):
  for quantization,valid in (("unknown",False),("right_shift_4_then_clip",True)):
   with self.subTest(quantization=quantization),tempfile.TemporaryDirectory() as d:
    p=Path(d)/"c.ini";p.write_text("[App]\nconfig_version=1.1\n[Data]\nsample_format=uint8\nadc_bit_width=12\nquantization="+quantization+"\n",encoding="utf8")
    if valid:self.assertEqual(load_config(p).data.signal_chain.adc_bit_width,12)
    else:
     with self.assertRaises(ConfigError):load_config(p)
 def test_raw_audit_defaults_disabled_for_missing_and_old_config(self):
  with tempfile.TemporaryDirectory() as d:
   missing=load_config(Path(d)/"missing.ini");self.assertFalse(missing.raw_audit.enabled);self.assertEqual(missing.raw_audit.retention_policy,"bounded_rotating_files")
   p=Path(d)/"old.ini";p.write_text("[App]\nconfig_version=1.1\n",encoding="utf8");old=load_config(p);self.assertFalse(old.raw_audit.enabled);self.assertEqual(old.config_version,"1.4")
 def test_raw_audit_explicit_enabled_config_is_exposed(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"c.ini";p.write_text("[App]\nconfig_version=1.2\n[RawAudit]\nenabled=true\nmax_bytes=1048576\nbackup_count=4\npayload_prefix_bytes=512\nrecord_valid=false\nrecord_invalid=true\nretention_policy=bounded_rotating_files\n",encoding="utf8")
   audit=load_config(p).raw_audit;self.assertIsInstance(audit,RawAuditConfig);self.assertTrue(audit.enabled);self.assertEqual((audit.max_bytes,audit.backup_count,audit.payload_prefix_bytes),(1048576,4,512));self.assertFalse(audit.record_valid);self.assertTrue(audit.record_invalid)
 def test_raw_audit_invalid_values_are_rejected(self):
  values=("enabled=maybe", "max_bytes=0", f"max_bytes={RAW_AUDIT_MAX_BYTES_LIMIT+1}", "backup_count=-1", "backup_count=101", "payload_prefix_bytes=-1", "payload_prefix_bytes=65537", "record_valid=maybe", "retention_policy=forever", "unknown_key=x")
  for value in values:
   with self.subTest(value=value),tempfile.TemporaryDirectory() as d:
    p=Path(d)/"c.ini";p.write_text("[App]\nconfig_version=1.2\n[RawAudit]\n"+value+"\n",encoding="utf8")
    with self.assertRaises(ConfigError):load_config(p)
  with self.assertRaises(ValueError):RawAuditConfig(enabled=True,record_valid=False,record_invalid=False)
 def test_raw_audit_control_characters_and_old_section_are_rejected(self):
  bodies=("[App]\nconfig_version=1.2\n[RawAudit]\nretention_policy=bounded\n continued\n", "[App]\nconfig_version=1.1\n[RawAudit]\nenabled=false\n")
  for body in bodies:
   with self.subTest(body=body),tempfile.TemporaryDirectory() as d:
    p=Path(d)/"c.ini";p.write_text(body,encoding="utf8")
    with self.assertRaises(ConfigError):load_config(p)
 def test_raw_audit_enabled_minimum_and_disabled_legacy_value(self):
  self.assertEqual(RawAuditConfig(enabled=True,max_bytes=MIN_RAW_AUDIT_FILE_BYTES).max_bytes,MIN_RAW_AUDIT_FILE_BYTES)
  self.assertEqual(RawAuditConfig(enabled=False,max_bytes=1).max_bytes,1)
  for value in (1,MIN_RAW_AUDIT_FILE_BYTES-1):
   with self.subTest(value=value),self.assertRaises(ValueError):RawAuditConfig(enabled=True,max_bytes=value)
 def test_raw_audit_total_budget_boundary_and_overflow(self):
  boundary=RawAuditConfig(max_bytes=RAW_AUDIT_MAX_BYTES_LIMIT,backup_count=1);self.assertEqual(boundary.total_budget_bytes,RAW_AUDIT_TOTAL_BUDGET_LIMIT)
  with self.assertRaises(ValueError):RawAuditConfig(max_bytes=RAW_AUDIT_MAX_BYTES_LIMIT,backup_count=2)
 def test_old_minor_enabled_undersized_audit_is_explicitly_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"c.ini";p.write_text("[App]\nconfig_version=1.2\n[RawAudit]\nenabled=true\nmax_bytes=1\n",encoding="utf8")
   with self.assertRaisesRegex(ConfigError,"at least"):load_config(p)
 def test_packet_protocol_defaults_to_strict_logical16_for_older_configs(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"c.ini";p.write_text("[App]\nconfig_version=1.3\n",encoding="utf8")
   protocol=load_config(p).packet_protocol
   self.assertEqual((protocol.mode,protocol.wire_packet_size,protocol.logical_packet_size,protocol.padding_rule),(LOGICAL16_PROTOCOL,16,16,"none"))
 def test_explicit_28_byte_protocol_is_strictly_validated(self):
  valid="[App]\nconfig_version=1.4\n[Protocol]\nmode=wire28_logical16_zero_suffix_v1\nwire_packet_size=28\nlogical_packet_size=16\npadding_rule=zero_suffix\nevidence_ref=probe.json\n"
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"c.ini";p.write_text(valid,encoding="utf8");self.assertEqual(load_config(p).packet_protocol.mode,PADDED28_PROTOCOL)
   for old,new in (("wire_packet_size=28","wire_packet_size=16"),("padding_rule=zero_suffix","padding_rule=none"),("evidence_ref=probe.json","evidence_ref=unknown"),("evidence_ref=probe.json","evidence_ref=built_in_logical16_contract")):
    with self.subTest(replacement=new):
     p.write_text(valid.replace(old,new),encoding="utf8")
     with self.assertRaises(ConfigError):load_config(p)
 def test_protocol_section_requires_version_1_4(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"c.ini";p.write_text("[App]\nconfig_version=1.3\n[Protocol]\nmode=logical16_odd_bytes_v1\n",encoding="utf8")
   with self.assertRaisesRegex(ConfigError,"requires config_version 1.4"):load_config(p)
 def test_valid_raw_audit_must_cover_configured_wire_packet(self):
  body="[App]\nconfig_version=1.4\n[Protocol]\nmode=wire28_logical16_zero_suffix_v1\nwire_packet_size=28\nlogical_packet_size=16\npadding_rule=zero_suffix\nevidence_ref=probe:sha256\n[RawAudit]\nenabled=true\nmax_bytes=8192\npayload_prefix_bytes=27\nrecord_valid=true\n"
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"c.ini";p.write_text(body,encoding="utf8")
   with self.assertRaisesRegex(ConfigError,"cover configured wire packet"):load_config(p)
 def test_invalid_only_raw_audit_has_same_wire_coverage_constraint(self):
  base="[App]\nconfig_version=1.4\n[Protocol]\nmode=wire28_logical16_zero_suffix_v1\nwire_packet_size=28\nlogical_packet_size=16\npadding_rule=zero_suffix\nevidence_ref=probe:sha256\n[RawAudit]\nenabled=true\nmax_bytes={max_bytes}\npayload_prefix_bytes={prefix}\nrecord_valid=false\nrecord_invalid=true\n"
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"c.ini"
   p.write_text(base.format(max_bytes=8192,prefix=27),encoding="utf8")
   with self.assertRaisesRegex(ConfigError,"cover configured wire packet"):load_config(p)
   p.write_text(base.format(max_bytes=8191,prefix=28),encoding="utf8")
   with self.assertRaisesRegex(ConfigError,"at least"):load_config(p)
   p.write_text(base.format(max_bytes=8192,prefix=28),encoding="utf8")
   self.assertEqual(load_config(p).raw_audit.max_bytes,8192)
if __name__=="__main__":unittest.main()
