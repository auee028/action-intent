# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: vintent.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='vintent.proto',
  package='action_module',
  syntax='proto3',
  serialized_pb=_b('\n\rvintent.proto\x12\raction_module\"8\n\nMsgRequest\x12\r\n\x05\x66rame\x18\x01 \x01(\x0c\x12\r\n\x05token\x18\x02 \x01(\t\x12\x0c\n\x04init\x18\x03 \x01(\x08\"\x8b\x01\n\x08MsgReply\x12\x0e\n\x06\x61\x63tion\x18\x01 \x01(\t\x12\x19\n\x11\x61\x63tion_confidence\x18\x02 \x01(\x02\x12\x1a\n\x12top3_intent_labels\x18\x03 \x01(\t\x12\x19\n\x11top3_intent_probs\x18\x04 \x01(\t\x12\x0e\n\x06object\x18\x05 \x01(\t\x12\r\n\x05\x62oxes\x18\x06 \x01(\t2L\n\x07Greeter\x12\x41\n\x07\x41nalyze\x12\x19.action_module.MsgRequest\x1a\x17.action_module.MsgReply\"\x00(\x01\x42\x08P\x01\xa2\x02\x03HLWb\x06proto3')
)




_MSGREQUEST = _descriptor.Descriptor(
  name='MsgRequest',
  full_name='action_module.MsgRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='frame', full_name='action_module.MsgRequest.frame', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='token', full_name='action_module.MsgRequest.token', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='init', full_name='action_module.MsgRequest.init', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=32,
  serialized_end=88,
)


_MSGREPLY = _descriptor.Descriptor(
  name='MsgReply',
  full_name='action_module.MsgReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='action', full_name='action_module.MsgReply.action', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='action_confidence', full_name='action_module.MsgReply.action_confidence', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='top3_intent_labels', full_name='action_module.MsgReply.top3_intent_labels', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='top3_intent_probs', full_name='action_module.MsgReply.top3_intent_probs', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='object', full_name='action_module.MsgReply.object', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='boxes', full_name='action_module.MsgReply.boxes', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=91,
  serialized_end=230,
)

DESCRIPTOR.message_types_by_name['MsgRequest'] = _MSGREQUEST
DESCRIPTOR.message_types_by_name['MsgReply'] = _MSGREPLY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

MsgRequest = _reflection.GeneratedProtocolMessageType('MsgRequest', (_message.Message,), dict(
  DESCRIPTOR = _MSGREQUEST,
  __module__ = 'vintent_pb2'
  # @@protoc_insertion_point(class_scope:action_module.MsgRequest)
  ))
_sym_db.RegisterMessage(MsgRequest)

MsgReply = _reflection.GeneratedProtocolMessageType('MsgReply', (_message.Message,), dict(
  DESCRIPTOR = _MSGREPLY,
  __module__ = 'vintent_pb2'
  # @@protoc_insertion_point(class_scope:action_module.MsgReply)
  ))
_sym_db.RegisterMessage(MsgReply)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('P\001\242\002\003HLW'))

_GREETER = _descriptor.ServiceDescriptor(
  name='Greeter',
  full_name='action_module.Greeter',
  file=DESCRIPTOR,
  index=0,
  options=None,
  serialized_start=232,
  serialized_end=308,
  methods=[
  _descriptor.MethodDescriptor(
    name='Analyze',
    full_name='action_module.Greeter.Analyze',
    index=0,
    containing_service=None,
    input_type=_MSGREQUEST,
    output_type=_MSGREPLY,
    options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_GREETER)

DESCRIPTOR.services_by_name['Greeter'] = _GREETER

# @@protoc_insertion_point(module_scope)
