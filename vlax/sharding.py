from dataclasses import dataclass

from scalax.sharding import (
    FSDPShardingRule,
    MeshShardingHelper,
    PartitionSpec,
    ShardingRule,
)

from vlax.models.big_vision.registry import Registry


@dataclass
class ShardingConfig:
    mesh: MeshShardingHelper
    model_sharding: ShardingRule
    data_sharding: ShardingRule


@Registry.register("sharding.fsdp")
def make_fsdp_sharding(model=-1, data=1) -> ShardingConfig:
    mesh = MeshShardingHelper([data, model], ["data", "fsdp"])
    return ShardingConfig(
        mesh=mesh,
        model_sharding=FSDPShardingRule(),
        data_sharding=PartitionSpec(("data", "fsdp")),
    )
