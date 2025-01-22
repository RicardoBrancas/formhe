def repair(instance, mcss, predicates_before_skip):
    solved = False
    first_depth = True
    for depth in range(config.get().minimum_depth, config.get().maximum_depth):
        for extra_statements in [[], [('empty', [])], [('empty', [None])], [('empty', [None]), ('empty', [None])]]:
            for mcs in mcss:
                modified_instance = Instance(config.get().input_file, skips=mcs, reference_instance=instance.reference)
                spec_generator = ASPSpecGenerator(modified_instance, modified_instance.config.extra_vars, predicates_before_skip.items())
                trinity_spec = spec_generator.trinity_spec
                asp_visitor = AspVisitor(trinity_spec, spec_generator.free_vars)
                preset_statements = []
                for rule in modified_instance.constantCollector.skipped:
                    preset_statements.append(asp_visitor.visit(rule))
                asp_interpreter = AspInterpreter(modified_instance)

                # runhelper.timer_start('answer.set.enum.time')
                # runhelper.timer_start('answer.set.enum.time.2')
                # if not config.get().no_semantic_constraints:
                #     modified_instance.find_wrong_models(max_sols=1000)
                # runhelper.timer_stop('answer.set.enum.time')
                # runhelper.timer_stop('answer.set.enum.time.2')

                # sorted_cores = sorted(modified_instance.cores, key=len)

                atom_enum_constructor = lambda p: Z3Enumerator(trinity_spec, depth, predicates_names=modified_instance.constantCollector.predicates.keys(), cores=None, free_vars=asp_visitor.free_vars,
                                                               preset_statements=list(p), strict_minimum_depth=not first_depth,
                                                               free_predicates=OrderedSet(predicates_before_skip.keys()) - modified_instance.constantCollector.predicates_generated,
                                                               force_generate_predicates=modified_instance.constantCollector.predicates_used - modified_instance.constantCollector.predicates_generated)
                statement_enumerator = StatementEnumerator(atom_enum_constructor, preset_statements + extra_statements, 1, asp_visitor.free_vars, depth)

                logger.info('Unsupported predicates: %s', str(set(OrderedSet(predicates_before_skip.keys()) - modified_instance.constantCollector.predicates_generated)))
                logger.info('Needed predicates: %s', str(set(modified_instance.constantCollector.predicates_used - modified_instance.constantCollector.predicates_generated)))

                runhelper.timer_start('eval.fail.time')
                while prog := next(statement_enumerator):
                    runhelper.tag_increment('enum.programs')
                    runhelper.timer_start('eval.time')
                    try:
                        asp_prog = asp_interpreter.eval(prog)
                        # print(asp_prog)
                        # logger.debug(prog)
                        # logger.debug(asp_prog)

                        res = asp_interpreter.test(asp_prog)
                        if res:
                            logger.info('Solution found')
                            runhelper.log_any('solution', asp_prog)

                            print('**Fix Suggestion**\n')

                            if mcs:
                                print(f'You can try replacing the following line{"s" if len(statement_enumerator.current_preset_statements) > 1 else ""}:\n')
                                print('\n'.join(['\t' + str(line) for line in mcs]))
                                print('\nWith (the "?" are missing parts you should fill in):\n')
                                print('\n'.join(['\t' + str(line) for line in statement_enumerator.current_preset_statements]))
                                print()
                            else:
                                print(f'You can try adding the following line{"s" if len(statement_enumerator.current_preset_statements) > 1 else ""} (the "?" are missing parts you should fill in):\n')
                                print('\n'.join(['\t' + str(line) for line in statement_enumerator.current_preset_statements]))
                                print()

                            # print('Solution found')
                            # print(mcs)
                            # print(statement_enumerator.current_preset_statements)
                            # print(asp_prog)
                            solved = True
                            break
                    except (RuntimeError, InstanceGroundingException) as e:
                        runhelper.timer_stop('eval.fail.time')
                        runhelper.tag_increment('eval.fail.programs')
                        logger.warning('Failed to parse: %s', prog)
                        # traceback.print_exception(e)
                        # exit()
                    except Exception as e:
                        # traceback.print_exception(e)
                        raise e
                    runhelper.timer_stop('eval.time')
                    runhelper.timer_start('eval.fail.time')

                if solved:
                    break
            if solved:
                break
        if solved:
            break

        first_depth = False
    if not solved:
        print('Synthesis Failed')
        exit(-2)