module ServerUtils

export get_server_config, ServerConfig

struct ServerConfig
    host::String
    port::Int
    base_url::String
end

function get_server_config(verbose = true)
    host = ENV["LOCAL_HOST"]
    port = parse(Int, ENV["SERVER_JL_PORT"])

    base_url = "http://$host:$port"

    if verbose
        println("Server configuration: $base_url")
    end

    return ServerConfig(host, port, base_url)
end

end # module
