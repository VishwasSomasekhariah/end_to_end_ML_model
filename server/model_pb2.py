# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: model.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0bmodel.proto\x12\x05model\"&\n\x0cmodelRequest\x12\x16\n\x0eprocessedImage\x18\x01 \x01(\x0c\"7\n\x12predictionResponse\x12\r\n\x05guess\x18\x01 \x01(\x05\x12\x12\n\nconfidence\x18\x02 \x01(\x02\x32M\n\tPredictor\x12@\n\x0cpredictImage\x12\x13.model.modelRequest\x1a\x19.model.predictionResponse\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'model_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _MODELREQUEST._serialized_start=22
  _MODELREQUEST._serialized_end=60
  _PREDICTIONRESPONSE._serialized_start=62
  _PREDICTIONRESPONSE._serialized_end=117
  _PREDICTOR._serialized_start=119
  _PREDICTOR._serialized_end=196
# @@protoc_insertion_point(module_scope)
