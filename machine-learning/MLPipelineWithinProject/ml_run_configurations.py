import subprocess
import gc
from ml_utils import collect_available_choices


def get_configurations():
    to_run = []
    choices = collect_available_choices()
    # these are your single projects
    choices["data"] = ["graphhopper-graphhopper","itext-itext7","fabric8io-kubernetes-client","apache-iotdb",
                       "nationalsecurityagency-emissary","apache-pulsar","wojciechzankowski-iextrading4j",
                       "seleniumhq-htmlunit-driver","cmu-phil-tetrad","questdb-questdb","logic-ng-logicng",
                       "opencb-opencga","geonetwork-core-geonetwork","zanata-zanata-platform",
                       "googleapis-google-http-java-client","finraos-herd","microsoft-azure-maven-plugins",
                       "instancio-instancio","arangodb-arangodb-java-driver","codestory-fluent-http"]

    with open('configuration.txt', 'r') as f:
        for line in f.readlines():
            params = {}
            conf = line.strip().split(" ")
            # default
            for key in choices.keys():
                params[key] = "none"
            params["k"] = 10
            params["feature_sel"] = "vif"

            for c in conf:
                for key in choices.keys():
                    if c in choices[key]:
                        params[key] = c
            to_run.append(params)
    return to_run


# run
for conf in get_configurations():
    subprocess.run(["python", "./ml_main.py", "-i", conf["data"], "-k", str(conf["k"]), "-p", conf["feature_sel"],
                    conf["balancing"], conf["optimization"], conf["classifier"]])
    gc.collect()


