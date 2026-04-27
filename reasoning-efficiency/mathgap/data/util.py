from typing import Dict, List, Tuple
import os
import json

import pandas as pd

from mathgap.natlang.templates import Template, TemplateParser, TemplateWithMetadataParser, TemplateType, TemplateCatalog
from mathgap.logicalforms import LogicalForm, Container, Transfer, CompEq, Comp, PartWhole, Rate

DATA_FOLDER = os.path.join("mathgap", "data")

TEMPLATE_TYPES = {
    Container: "container",
    Transfer: "transfer",
    Comp: "comp",
    CompEq: "compeq",
    PartWhole: "partwhole",
    Rate: "rate"
}

def load_agents(data_folder: str = DATA_FOLDER, version: str = "v1") -> List[str]:
    return list(pd.read_csv(os.path.join(data_folder, "agents", version, "agents.csv"), index_col=False, names=["name"])["name"])

def load_attributes(data_folder: str = DATA_FOLDER, version: str = "v1") -> List[str]:
    return list(pd.read_csv(os.path.join(data_folder, "attributes", version, "attributes.csv"), index_col=False, names=["attr"])["attr"])

def load_entities(data_folder: str = DATA_FOLDER, version: str = "v1") -> Dict[str, List[str] | Dict[str, str]]:
    with open(os.path.join(data_folder, "entities", version, "entities.json"), "r") as f:
        entities = [tuple(e) for e in json.load(f)]
    with open(os.path.join(data_folder, "entities", version, "entity_unit_names.json"), "r") as f:
        unit_by_entity = json.load(f)
    with open(os.path.join(data_folder, "entities", version, "entities_part_whole.json"), "r") as f:
        entities_part_whole = {tuple(phdata["whole"]): [tuple(p) for p in phdata["parts"]] for phdata in json.load(f)}
    with open(os.path.join(data_folder, "entities", version, "entities_rate.json"), "r") as f:
        super_sub_entities = {tuple(rdata["super"]): [tuple(s) for s in rdata["sub"]] for rdata in json.load(f)}

    return {
        "entities_without_units": entities,
        "entities_with_units": unit_by_entity,
        "parts_by_whole": entities_part_whole,
        "super_sub_entities": super_sub_entities
    }

def load_templates(data_folder: str = DATA_FOLDER, template_parser: TemplateParser = None, version: str = "v1") -> TemplateCatalog:
    """ Loads all natural-language templates of a specific version """
    if template_parser is None: template_parser = TemplateWithMetadataParser()

    def load_template_for_lf(lf_name):
        template_file = os.path.join(data_folder, "templates", version, f"{lf_name}.json")
        if not os.path.exists(template_file): return None

        with open(template_file, "r") as f:
            return template_parser.parse(json.load(f))

    templates_by_lf_and_type = {}
    for t,n in TEMPLATE_TYPES.items():
        t_for_lf = load_template_for_lf(n)
        if t_for_lf is not None:
            templates_by_lf_and_type[t] = t_for_lf
    return TemplateCatalog(templates_by_lf_and_type)
