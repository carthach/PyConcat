def buildNetworkGraph(costMatrix):
    import json

    data = {}

    nodes = []
    links = []

    for nodeIndex, nodeConnections in enumerate(costMatrix):
        node = {}

        node["id"] = str(nodeIndex)
        node["group"] = nodeIndex

        nodes.append(node)

        for nodeConnectionIndex, nodeConnection in enumerate(nodeConnections):
            connection = {}

            connection["source"] = str(nodeIndex)
            connection["target"] = str(nodeConnectionIndex)
            connection["value"] = nodeConnection

            links.append(connection)

    data["nodes"] = nodes
    data["links"] = links

    jsonString = json.dumps(data, sort_keys=True, indent = 4, separators = (',', ': '))

    return jsonString

def createD3Diagram(costMatrix, outputPath):
    import os
    import shutil

    graphOutputPath = outputPath + "/graphs"

    if not os.path.exists(graphOutputPath):
        os.mkdir(graphOutputPath)

    shutil.copy2("d3/d3.min.js", graphOutputPath)
    shutil.copy2("d3/index.html", graphOutputPath)

    outputFile = graphOutputPath + '/data.json'

    f = open(outputFile, 'w')

    f.write(buildNetworkGraph(costMatrix))









