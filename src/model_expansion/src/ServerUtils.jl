module ServerUtils

export get_server_config, ServerConfig

struct ServerConfig
    host::String
    port::Int
    base_url::String
end

function get_server_config(default_host = "localhost", default_port = 8081; verbose = true)
    host = default_host

    port = if length(ARGS) >= 1
        parse(Int, ARGS[1])
    elseif haskey(ENV, "SERVER_PORT")
        parse(Int, ENV["SERVER_PORT"])
    else
        default_port
    end

    base_url = "http://$host:$port"

    if verbose
        println("Server configuration: $base_url")
    end

    return ServerConfig(host, port, base_url)
end

end # module
